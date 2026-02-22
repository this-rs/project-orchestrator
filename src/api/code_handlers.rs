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
    /// Filter by project slug (takes precedence over workspace_slug)
    pub project_slug: Option<String>,
    /// Filter by workspace slug (searches all projects in the workspace)
    pub workspace_slug: Option<String>,
}

/// Search code semantically across the codebase
///
/// Returns `SearchHit<CodeDocument>` directly so the frontend gets
/// the `{ document, score }` shape it expects.
///
/// When `workspace_slug` is provided (and `project_slug` is not), searches all
/// projects in the workspace, merges results by score, and truncates to `limit`.
pub async fn search_code(
    State(state): State<OrchestratorState>,
    Query(params): Query<CodeSearchQuery>,
) -> Result<
    Json<Vec<crate::meilisearch::indexes::SearchHit<crate::meilisearch::indexes::CodeDocument>>>,
    AppError,
> {
    let limit = params.limit.unwrap_or(10);

    // If project_slug is given, use it directly (backward compat)
    if params.project_slug.is_some() {
        let hits = state
            .orchestrator
            .meili()
            .search_code_with_scores(
                &params.query,
                limit,
                params.language.as_deref(),
                params.project_slug.as_deref(),
                None,
            )
            .await?;
        return Ok(Json(hits));
    }

    // If workspace_slug is given, resolve to project slugs and merge results
    if let Some(ref ws_slug) = params.workspace_slug {
        let workspace = state
            .orchestrator
            .neo4j()
            .get_workspace_by_slug(ws_slug)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Workspace not found: {}", ws_slug)))?;

        let projects = state
            .orchestrator
            .neo4j()
            .list_workspace_projects(workspace.id)
            .await?;

        let mut all_hits = Vec::new();
        for project in &projects {
            let hits = state
                .orchestrator
                .meili()
                .search_code_with_scores(
                    &params.query,
                    limit,
                    params.language.as_deref(),
                    Some(&project.slug),
                    None,
                )
                .await
                .unwrap_or_default();
            all_hits.extend(hits);
        }

        // Sort by score descending and truncate to limit
        all_hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hits.truncate(limit);

        return Ok(Json(all_hits));
    }

    // No filter — global search
    let hits = state
        .orchestrator
        .meili()
        .search_code_with_scores(&params.query, limit, params.language.as_deref(), None, None)
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
    /// Community labels affected by this change (from graph analytics)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub affected_communities: Vec<String>,
    /// Betweenness centrality of the target node (bridge score)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub betweenness_score: Option<f64>,
    /// Human-readable explanation of the risk score computation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_formula: Option<String>,
}

/// Analyze impact of changing a file or function
pub async fn analyze_impact(
    State(state): State<OrchestratorState>,
    Query(query): Query<ImpactQuery>,
) -> Result<Json<ImpactAnalysis>, AppError> {
    let target_type = query.target_type.as_deref().unwrap_or("file");

    let (project_id, project_root) = if let Some(ref slug) = query.project_slug {
        let project = state
            .orchestrator
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;
        (Some(project.id), Some(project.root_path))
    } else {
        (None, None)
    };

    // Resolve relative file paths to absolute using project root_path
    let target = if target_type == "file" && !query.target.starts_with('/') {
        if let Some(ref root) = project_root {
            let expanded = crate::expand_tilde(root);
            format!("{}/{}", expanded.trim_end_matches('/'), &query.target)
        } else {
            query.target.clone()
        }
    } else {
        query.target.clone()
    };

    let (directly_affected, transitively_affected, caller_count, target) = if target_type == "file"
    {
        let direct = state
            .orchestrator
            .neo4j()
            .find_impacted_files(&target, 1, project_id)
            .await?;
        // Fallback: if resolved path found nothing, retry with the raw input
        let (direct, effective) = if direct.is_empty() && target != query.target {
            let fallback = state
                .orchestrator
                .neo4j()
                .find_impacted_files(&query.target, 1, project_id)
                .await?;
            if !fallback.is_empty() {
                (fallback, query.target.clone())
            } else {
                (direct, target)
            }
        } else {
            (direct, target)
        };
        let transitive = state
            .orchestrator
            .neo4j()
            .find_impacted_files(&effective, 3, project_id)
            .await?;
        let count = direct.len() as i64;
        (direct, transitive, count, effective)
    } else {
        let callers = state
            .orchestrator
            .neo4j()
            .find_callers(&target, project_id)
            .await?;
        let direct: Vec<String> = callers.iter().map(|f| f.file_path.clone()).collect();
        let count = state
            .orchestrator
            .neo4j()
            .get_function_caller_count(&target, project_id)
            .await?;
        (direct.clone(), direct, count, target)
    };

    let test_files: Vec<String> = transitively_affected
        .iter()
        .filter(|p| p.contains("test") || p.contains("_test") || p.ends_with("_tests.rs"))
        .cloned()
        .collect();

    // Fetch GDS analytics for the target node (best-effort: None if not computed)
    // Try resolved path first, fall back to raw input if analytics not found
    let node_analytics = {
        let analytics = state
            .orchestrator
            .neo4j()
            .get_node_analytics(&target, target_type)
            .await
            .unwrap_or(None);
        match analytics {
            Some(a) => Some(a),
            None if target != query.target => state
                .orchestrator
                .neo4j()
                .get_node_analytics(&query.target, target_type)
                .await
                .unwrap_or(None),
            None => None,
        }
    };

    // Collect all affected file paths for community analysis
    let all_affected: Vec<String> = directly_affected
        .iter()
        .chain(transitively_affected.iter())
        .cloned()
        .collect();
    let affected_communities = state
        .orchestrator
        .neo4j()
        .get_affected_communities(&all_affected)
        .await
        .unwrap_or_default();

    // Compute risk level using composite formula when GDS data is available
    let (risk_level, betweenness_score, risk_formula) = if let Some(ref analytics) = node_analytics
    {
        if let Some(betweenness) = analytics.betweenness {
            // Composite risk formula:
            // betweenness_score = clamp(betweenness * 3.0, 0, 1)
            // community_spread = affected_communities / total (use 5 as reasonable estimate)
            // degree_score = clamp(caller_count / 20.0, 0, 1)
            // risk = betweenness_score * 0.5 + community_spread * 0.3 + degree_score * 0.2
            let bs = (betweenness * 3.0).clamp(0.0, 1.0);
            let total_communities = 5.0_f64; // reasonable default
            let cs = (affected_communities.len() as f64 / total_communities).clamp(0.0, 1.0);
            let ds = (caller_count as f64 / 20.0).clamp(0.0, 1.0);
            let risk_score = bs * 0.5 + cs * 0.3 + ds * 0.2;

            let level = if risk_score > 0.7 {
                "high"
            } else if risk_score > 0.3 {
                "medium"
            } else {
                "low"
            };

            let formula = format!(
                "betweenness={:.2} ({}), {}/{} communities affected, {} callers → score={:.2}",
                betweenness,
                if betweenness > 0.23 {
                    "high bridge"
                } else if betweenness > 0.1 {
                    "moderate bridge"
                } else {
                    "low bridge"
                },
                affected_communities.len(),
                total_communities as usize,
                caller_count,
                risk_score,
            );

            (level.to_string(), Some(betweenness), Some(formula))
        } else {
            // Analytics exist but no betweenness → fallback
            let level = if transitively_affected.len() > 10 || caller_count > 10 {
                "high"
            } else if transitively_affected.len() > 3 || caller_count > 3 {
                "medium"
            } else {
                "low"
            };
            (level.to_string(), None, None)
        }
    } else {
        // No GDS analytics at all → fallback to legacy threshold logic
        let level = if transitively_affected.len() > 10 || caller_count > 10 {
            "high"
        } else if transitively_affected.len() > 3 || caller_count > 3 {
            "medium"
        } else {
            "low"
        };
        (level.to_string(), None, None)
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
        target,
        directly_affected,
        transitively_affected,
        test_files_affected: test_files,
        caller_count,
        risk_level,
        suggestion,
        affected_communities,
        betweenness_score,
        risk_formula,
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
    /// Detected code communities from graph analytics (empty if analytics not yet computed)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub communities: Vec<CommunityOverview>,
}

/// Summary of a detected code community (Louvain clustering)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityOverview {
    /// Numeric community identifier
    pub id: i64,
    /// Human-readable community label
    pub label: String,
    /// Number of files in this community
    pub file_count: usize,
    /// Top files in this community (by pagerank)
    pub key_files: Vec<String>,
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

#[derive(Deserialize)]
pub struct ArchitectureQuery {
    /// Filter by project slug (takes precedence over workspace_slug)
    pub project_slug: Option<String>,
    /// Filter by workspace slug (aggregates architecture of all workspace projects)
    pub workspace_slug: Option<String>,
}

/// Get high-level architecture overview
///
/// When `workspace_slug` is provided (and `project_slug` is not), aggregates
/// language stats and key files across all projects in the workspace.
pub async fn get_architecture(
    State(state): State<OrchestratorState>,
    Query(params): Query<ArchitectureQuery>,
) -> Result<Json<ArchitectureOverview>, AppError> {
    // Resolve project_ids: single project, workspace projects, or global
    let project_ids: Option<Vec<uuid::Uuid>> = if let Some(ref slug) = params.project_slug {
        // Single project filter (backward compat)
        let project = state
            .orchestrator
            .neo4j()
            .get_project_by_slug(slug)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;
        Some(vec![project.id])
    } else if let Some(ref ws_slug) = params.workspace_slug {
        // Workspace filter: resolve to all project IDs
        let workspace = state
            .orchestrator
            .neo4j()
            .get_workspace_by_slug(ws_slug)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Workspace not found: {}", ws_slug)))?;
        let projects = state
            .orchestrator
            .neo4j()
            .list_workspace_projects(workspace.id)
            .await?;
        Some(projects.into_iter().map(|p| p.id).collect())
    } else {
        None
    };

    // Aggregate language stats
    let lang_stats = match &project_ids {
        Some(ids) => {
            let mut all_stats = Vec::new();
            for pid in ids {
                let stats = state
                    .orchestrator
                    .neo4j()
                    .get_language_stats_for_project(*pid)
                    .await?;
                all_stats.extend(stats);
            }
            // Merge stats by language
            let mut merged: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for s in all_stats {
                *merged.entry(s.language).or_default() += s.file_count;
            }
            merged
                .into_iter()
                .map(
                    |(language, file_count)| crate::neo4j::models::LanguageStatsNode {
                        language,
                        file_count,
                    },
                )
                .collect::<Vec<_>>()
        }
        None => state.orchestrator.neo4j().get_language_stats().await?,
    };

    let languages: Vec<LanguageStats> = lang_stats
        .into_iter()
        .map(|s| LanguageStats {
            language: s.language,
            file_count: s.file_count,
            function_count: 0,
            struct_count: 0,
        })
        .collect();

    // Aggregate key files
    let key_files = match &project_ids {
        Some(ids) => {
            let mut all_files = Vec::new();
            for pid in ids {
                let files = state
                    .orchestrator
                    .neo4j()
                    .get_most_connected_files_for_project(*pid, 10)
                    .await?;
                all_files.extend(files);
            }
            // Sort by pagerank (descending) with fallback to dependents
            all_files.sort_by(|a, b| {
                let pr_a = a.pagerank.unwrap_or(0.0);
                let pr_b = b.pagerank.unwrap_or(0.0);
                pr_b.partial_cmp(&pr_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.dependents.cmp(&a.dependents))
            });
            all_files.truncate(10);
            all_files
        }
        None => {
            state
                .orchestrator
                .neo4j()
                .get_most_connected_files_detailed(10)
                .await?
        }
    };

    // Aggregate community overviews from graph analytics
    let communities: Vec<CommunityOverview> = match &project_ids {
        Some(ids) => {
            let mut all_rows = Vec::new();
            for pid in ids {
                let comms = state
                    .orchestrator
                    .neo4j()
                    .get_project_communities(*pid)
                    .await?;
                all_rows.extend(comms);
            }
            all_rows
                .into_iter()
                .map(|r| CommunityOverview {
                    id: r.community_id,
                    label: r.community_label,
                    file_count: r.file_count,
                    key_files: r.key_files,
                })
                .collect()
        }
        None => vec![],
    };

    let total_files: usize = languages.iter().map(|l| l.file_count).sum();

    Ok(Json(ArchitectureOverview {
        total_files,
        languages,
        modules: vec![],
        key_files,
        orphan_files: vec![],
        communities,
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
    pub filter_community: Option<bool>,
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
        entity_count: None,
        entry_function: None,
        build_depth: None,
        include_relations: None,
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
                "entity_count": fg.entity_count,
                "entry_function": fg.entry_function,
                "build_depth": fg.build_depth,
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
            body.filter_community,
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

// ============================================================================
// Structural Analytics
// ============================================================================

#[derive(Deserialize)]
pub struct CommunitiesQuery {
    pub project_slug: String,
    pub min_size: Option<usize>,
}

/// GET /api/code/communities
pub async fn get_code_communities(
    State(state): State<OrchestratorState>,
    Query(params): Query<CommunitiesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let min_size = params.min_size.unwrap_or(2);
    let communities = state
        .orchestrator
        .neo4j()
        .get_project_communities(project.id)
        .await?;

    if communities.is_empty() {
        return Ok(Json(serde_json::json!({
            "message": "No structural analytics available. Run sync_project first.",
            "communities": [],
            "total_files": 0
        })));
    }

    let mut filtered: Vec<_> = communities
        .into_iter()
        .filter(|c| c.file_count >= min_size)
        .collect();
    filtered.sort_by(|a, b| b.file_count.cmp(&a.file_count));

    let total_files: usize = filtered.iter().map(|c| c.file_count).sum();

    let communities_json: Vec<serde_json::Value> = filtered
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.community_id,
                "label": c.community_label,
                "size": c.file_count,
                "key_files": c.key_files,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "communities": communities_json,
        "total_files": total_files,
        "community_count": communities_json.len(),
    })))
}

#[derive(Deserialize)]
pub struct CodeHealthQuery {
    pub project_slug: String,
    pub god_function_threshold: Option<usize>,
}

/// GET /api/code/health
pub async fn get_code_health(
    State(state): State<OrchestratorState>,
    Query(params): Query<CodeHealthQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let threshold = params.god_function_threshold.unwrap_or(10);
    let report = state
        .orchestrator
        .neo4j()
        .get_code_health_report(project.id, threshold)
        .await?;

    let circular_deps = state
        .orchestrator
        .neo4j()
        .get_circular_dependencies(project.id)
        .await?;

    let god_functions_json: Vec<serde_json::Value> = report
        .god_functions
        .iter()
        .map(|g| {
            serde_json::json!({
                "name": g.name,
                "file": g.file,
                "in_degree": g.in_degree,
                "out_degree": g.out_degree,
            })
        })
        .collect();

    let coupling_json = report.coupling_metrics.as_ref().map(|c| {
        serde_json::json!({
            "avg_clustering_coefficient": c.avg_clustering_coefficient,
            "max_clustering_coefficient": c.max_clustering_coefficient,
            "most_coupled_file": c.most_coupled_file,
        })
    });

    Ok(Json(serde_json::json!({
        "god_functions": god_functions_json,
        "god_function_count": god_functions_json.len(),
        "god_function_threshold": threshold,
        "orphan_files": report.orphan_files,
        "orphan_file_count": report.orphan_files.len(),
        "coupling_metrics": coupling_json,
        "circular_dependencies": circular_deps,
        "circular_dependency_count": circular_deps.len(),
    })))
}

// ============================================================================
// Implementation Planner
// ============================================================================

#[derive(Deserialize)]
pub struct PlanImplementationBody {
    pub project_slug: String,
    pub description: String,
    #[serde(default)]
    pub entry_points: Option<Vec<String>>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub auto_create_plan: Option<bool>,
}

pub async fn plan_implementation(
    State(state): State<OrchestratorState>,
    Json(body): Json<PlanImplementationBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::orchestrator::planner::{PlanRequest, PlanScope};

    // Resolve project slug → project ID
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Project '{}' not found", body.project_slug))?;

    let scope: Option<PlanScope> = body
        .scope
        .as_deref()
        .and_then(|s| serde_json::from_str(&format!("\"{}\"", s)).ok());

    let request = PlanRequest {
        project_id: project.id,
        project_slug: Some(body.project_slug),
        description: body.description,
        entry_points: body.entry_points,
        scope,
        auto_create_plan: body.auto_create_plan,
        root_path: Some(project.root_path),
    };

    let plan = state
        .orchestrator
        .planner()
        .plan_implementation(request)
        .await?;

    Ok(Json(
        serde_json::to_value(&plan).map_err(anyhow::Error::from)?,
    ))
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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
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

    #[test]
    fn test_code_search_query_with_workspace_slug() {
        let json = r#"{"query":"test","workspace_slug":"my-ws"}"#;
        let q: CodeSearchQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "test");
        assert_eq!(q.workspace_slug.as_deref(), Some("my-ws"));
        assert!(q.project_slug.is_none());
    }

    #[test]
    fn test_architecture_query_with_workspace_slug() {
        let json = r#"{"workspace_slug":"my-ws"}"#;
        let q: ArchitectureQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.workspace_slug.as_deref(), Some("my-ws"));
        assert!(q.project_slug.is_none());
    }

    /// Build a test router with a workspace containing a project with seeded code
    async fn test_app_with_workspace() -> axum::Router {
        use crate::neo4j::models::{ProjectNode, WorkspaceNode};

        let app_state = mock_app_state();

        // Create workspace
        let ws_id = uuid::Uuid::new_v4();
        let workspace = WorkspaceNode {
            id: ws_id,
            name: "Test Workspace".to_string(),
            slug: "test-ws".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            updated_at: None,
            metadata: serde_json::Value::Null,
        };
        app_state.neo4j.create_workspace(&workspace).await.unwrap();

        // Create project and link to workspace
        let proj_id = uuid::Uuid::new_v4();
        let project = ProjectNode {
            id: proj_id,
            name: "Test Project".to_string(),
            slug: "test-project".to_string(),
            root_path: "/tmp/test".to_string(),
            description: None,
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
        };
        app_state.neo4j.create_project(&project).await.unwrap();
        app_state
            .neo4j
            .add_project_to_workspace(ws_id, proj_id)
            .await
            .unwrap();

        // Seed code document
        let doc = CodeDocument {
            id: "src/main.rs".to_string(),
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            symbols: vec!["main".to_string()],
            docstrings: "Entry point".to_string(),
            signatures: vec!["fn main()".to_string()],
            imports: vec![],
            project_id: proj_id.to_string(),
            project_slug: "test-project".to_string(),
        };
        app_state.meili.index_code(&doc).await.unwrap();

        // Seed file for architecture
        let file = FileNode {
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(proj_id),
        };
        app_state.neo4j.upsert_file(&file).await.unwrap();

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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        create_router(state)
    }

    #[tokio::test]
    async fn test_search_code_with_workspace_slug_filter() {
        let app = test_app_with_workspace().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/search?query=main&workspace_slug=test-ws",
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
        assert_eq!(results[0]["document"]["path"], "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_code_workspace_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/search?query=main&workspace_slug=nonexistent",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_architecture_with_workspace_slug() {
        let app = test_app_with_workspace().await;
        let resp = app
            .oneshot(auth_get("/api/code/architecture?workspace_slug=test-ws"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total_files"], 1);
        let languages = json["languages"].as_array().unwrap();
        assert_eq!(languages.len(), 1);
        assert_eq!(languages[0]["language"], "rust");
    }

    #[tokio::test]
    async fn test_architecture_workspace_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/architecture?workspace_slug=nonexistent",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
            analytics_computed_at: None,
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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
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

    // ====================================================================
    // GET /api/code/architecture — GDS enrichment tests (Task 3.1)
    // ====================================================================

    /// Build a test router with files that have GDS analytics scores
    async fn test_app_with_analytics() -> (axum::Router, uuid::Uuid) {
        use crate::graph::models::FileAnalyticsUpdate;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::ProjectNode;
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();
        let project_id = uuid::Uuid::new_v4();

        // Create project
        let project = ProjectNode {
            id: project_id,
            name: "analytics-proj".to_string(),
            slug: "analytics-proj".to_string(),
            description: None,
            root_path: "/tmp/analytics".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: Some(chrono::Utc::now()),
            analytics_computed_at: None,
        };
        graph.create_project(&project).await.unwrap();

        // Seed files with project_id
        let file_paths = [
            "src/main.rs",
            "src/handler.rs",
            "src/utils.rs",
            "src/config.rs",
        ];
        for path in &file_paths {
            let file = FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "h".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project_id),
            };
            graph.upsert_file(&file).await.unwrap();
        }

        // Seed import relationships
        // main.rs imports handler.rs and config.rs
        graph
            .create_import_relationship("src/main.rs", "src/handler.rs", "handler")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/main.rs", "src/config.rs", "config")
            .await
            .unwrap();
        // handler.rs imports utils.rs
        graph
            .create_import_relationship("src/handler.rs", "src/utils.rs", "utils")
            .await
            .unwrap();

        // Seed GDS analytics: main.rs has HIGH pagerank, utils.rs has LOW pagerank
        let analytics = vec![
            FileAnalyticsUpdate {
                path: "src/main.rs".to_string(),
                pagerank: 0.15,
                betweenness: 0.30,
                community_id: 0,
                community_label: "Core".to_string(),
                clustering_coefficient: 0.5,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/handler.rs".to_string(),
                pagerank: 0.10,
                betweenness: 0.20,
                community_id: 0,
                community_label: "Core".to_string(),
                clustering_coefficient: 0.3,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/utils.rs".to_string(),
                pagerank: 0.05,
                betweenness: 0.05,
                community_id: 1,
                community_label: "Utilities".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/config.rs".to_string(),
                pagerank: 0.03,
                betweenness: 0.01,
                community_id: 1,
                community_label: "Utilities".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
        ];
        graph.batch_update_file_analytics(&analytics).await.unwrap();

        let app_state = mock_app_state_with(graph, MockSearchStore::new());
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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        (create_router(state), project_id)
    }

    #[tokio::test]
    async fn test_architecture_with_gds_scores() {
        // Test 1: Files with GDS analytics return pagerank, community fields
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/architecture?project_slug=analytics-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let key_files = json["key_files"].as_array().unwrap();
        assert!(!key_files.is_empty());

        // The first file should be src/main.rs (highest pagerank = 0.15)
        assert_eq!(key_files[0]["path"], "src/main.rs");
        assert!(key_files[0]["pagerank"].as_f64().unwrap() > 0.14);
        assert!(key_files[0]["betweenness"].as_f64().unwrap() > 0.0);
        assert_eq!(key_files[0]["community_label"], "Core");

        // Verify communities are present
        let communities = json["communities"].as_array().unwrap();
        assert_eq!(
            communities.len(),
            2,
            "Should have 2 communities: Core and Utilities"
        );

        let core = communities
            .iter()
            .find(|c| c["label"] == "Core")
            .expect("Core community should exist");
        assert_eq!(core["file_count"], 2);

        let utils = communities
            .iter()
            .find(|c| c["label"] == "Utilities")
            .expect("Utilities community should exist");
        assert_eq!(utils["file_count"], 2);
    }

    #[tokio::test]
    async fn test_architecture_without_gds_scores_fallback() {
        // Test 2: Files WITHOUT GDS analytics → same response as before (fallback to degree)
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

        // key_files should still work (may be empty if no import relationships seeded)
        assert!(json["key_files"].is_array());
        // communities should be absent (skip_serializing_if = Vec::is_empty)
        assert!(
            json.get("communities").is_none()
                || json["communities"].as_array().is_none_or(|c| c.is_empty()),
            "No communities should be returned when GDS not computed"
        );
    }

    #[tokio::test]
    async fn test_architecture_pagerank_ordering_overrides_degree() {
        // Test 3: File with high pagerank but lower degree should rank higher
        // than file with low pagerank but higher degree
        use crate::graph::models::FileAnalyticsUpdate;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::ProjectNode;
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();
        let project_id = uuid::Uuid::new_v4();

        let project = ProjectNode {
            id: project_id,
            name: "ordering-proj".to_string(),
            slug: "ordering-proj".to_string(),
            description: None,
            root_path: "/tmp/ordering".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: Some(chrono::Utc::now()),
            analytics_computed_at: None,
        };
        graph.create_project(&project).await.unwrap();

        // file_a: high degree (3 dependents) but LOW pagerank
        // file_b: low degree (1 dependent) but HIGH pagerank
        for path in &[
            "src/file_a.rs",
            "src/file_b.rs",
            "src/dep1.rs",
            "src/dep2.rs",
            "src/dep3.rs",
        ] {
            let file = FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "h".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project_id),
            };
            graph.upsert_file(&file).await.unwrap();
        }

        // file_a has 3 dependents (high degree)
        graph
            .create_import_relationship("src/dep1.rs", "src/file_a.rs", "file_a")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/dep2.rs", "src/file_a.rs", "file_a")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/dep3.rs", "src/file_a.rs", "file_a")
            .await
            .unwrap();

        // file_b has 1 dependent (low degree)
        graph
            .create_import_relationship("src/dep1.rs", "src/file_b.rs", "file_b")
            .await
            .unwrap();

        // But file_b has HIGHER pagerank than file_a
        let analytics = vec![
            FileAnalyticsUpdate {
                path: "src/file_a.rs".to_string(),
                pagerank: 0.05, // LOW pagerank despite high degree
                betweenness: 0.1,
                community_id: 0,
                community_label: "Main".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/file_b.rs".to_string(),
                pagerank: 0.15, // HIGH pagerank despite low degree
                betweenness: 0.3,
                community_id: 0,
                community_label: "Main".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
        ];
        graph.batch_update_file_analytics(&analytics).await.unwrap();

        let app_state = mock_app_state_with(graph, MockSearchStore::new());
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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get(
                "/api/code/architecture?project_slug=ordering-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let key_files = json["key_files"].as_array().unwrap();
        // file_b (pagerank 0.15) should be BEFORE file_a (pagerank 0.05)
        // even though file_a has more dependents
        let first_path = key_files[0]["path"].as_str().unwrap();
        assert_eq!(
            first_path, "src/file_b.rs",
            "File with higher pagerank should rank first, regardless of degree"
        );
    }

    // ====================================================================
    // GET /api/code/impact — GDS enrichment tests (Task 3.2)
    // ====================================================================

    /// Build a test router with import relationships and GDS analytics
    /// for testing the enriched impact analysis.
    async fn test_app_with_impact_analytics() -> axum::Router {
        use crate::graph::models::FileAnalyticsUpdate;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::ProjectNode;
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();
        let project_id = uuid::Uuid::new_v4();

        let project = ProjectNode {
            id: project_id,
            name: "impact-proj".to_string(),
            slug: "impact-proj".to_string(),
            description: None,
            root_path: "/tmp/impact".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: Some(chrono::Utc::now()),
            analytics_computed_at: None,
        };
        graph.create_project(&project).await.unwrap();

        // Barbell graph: cluster A ←→ bridge ←→ cluster B
        let all_files = [
            "src/bridge.rs", // bridge between clusters
            "src/a1.rs",     // cluster A
            "src/a2.rs",     // cluster A
            "src/b1.rs",     // cluster B
            "src/b2.rs",     // cluster B
            "src/leaf.rs",   // isolated leaf
        ];
        for path in &all_files {
            let file = FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "h".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project_id),
            };
            graph.upsert_file(&file).await.unwrap();
            graph
                .project_files
                .write()
                .await
                .entry(project_id)
                .or_default()
                .push(path.to_string());
        }

        // Import relationships: bridge connects both clusters
        // a1 → bridge, a2 → bridge (cluster A depends on bridge)
        // bridge → b1, bridge → b2 (bridge depends on cluster B)
        graph
            .create_import_relationship("src/a1.rs", "src/bridge.rs", "bridge")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/a2.rs", "src/bridge.rs", "bridge")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/bridge.rs", "src/b1.rs", "b1")
            .await
            .unwrap();
        graph
            .create_import_relationship("src/bridge.rs", "src/b2.rs", "b2")
            .await
            .unwrap();

        // leaf has no connections
        // (leaf.rs has no import relationships)

        // GDS analytics:
        // bridge.rs = high betweenness (0.4), community 0 ("Core")
        // a1, a2 = low betweenness, community 1 ("Cluster A")
        // b1, b2 = low betweenness, community 2 ("Cluster B")
        // leaf.rs = very low betweenness, community 3 ("Isolated")
        let analytics = vec![
            FileAnalyticsUpdate {
                path: "src/bridge.rs".to_string(),
                pagerank: 0.20,
                betweenness: 0.40,
                community_id: 0,
                community_label: "Core".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/a1.rs".to_string(),
                pagerank: 0.08,
                betweenness: 0.02,
                community_id: 1,
                community_label: "Cluster A".to_string(),
                clustering_coefficient: 0.5,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/a2.rs".to_string(),
                pagerank: 0.07,
                betweenness: 0.01,
                community_id: 1,
                community_label: "Cluster A".to_string(),
                clustering_coefficient: 0.5,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/b1.rs".to_string(),
                pagerank: 0.06,
                betweenness: 0.03,
                community_id: 2,
                community_label: "Cluster B".to_string(),
                clustering_coefficient: 0.5,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/b2.rs".to_string(),
                pagerank: 0.05,
                betweenness: 0.02,
                community_id: 2,
                community_label: "Cluster B".to_string(),
                clustering_coefficient: 0.5,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: "src/leaf.rs".to_string(),
                pagerank: 0.01,
                betweenness: 0.01,
                community_id: 3,
                community_label: "Isolated".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
        ];
        graph.batch_update_file_analytics(&analytics).await.unwrap();

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
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        });
        create_router(state)
    }

    #[tokio::test]
    async fn test_impact_bridge_file_high_risk() {
        // Test 1: Bridge file with high betweenness → risk should be elevated
        let app = test_app_with_impact_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/impact?target=src/bridge.rs&project_slug=impact-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Bridge has betweenness 0.4 → risk_formula should be present
        assert!(json["betweenness_score"].as_f64().unwrap() > 0.3);
        assert!(json["risk_formula"].is_string());
        let formula = json["risk_formula"].as_str().unwrap();
        assert!(formula.contains("high bridge"), "Formula: {}", formula);

        // Affected communities: dependents of bridge are a1, a2 (Cluster A)
        // find_dependent_files returns files that depend ON the target
        let communities = json["affected_communities"].as_array().unwrap();
        assert!(
            !communities.is_empty(),
            "Bridge should have affected communities from its dependents"
        );
        assert!(
            communities.iter().any(|c| c == "Cluster A"),
            "Cluster A files depend on bridge, got: {:?}",
            communities
        );

        // Key check: risk should be elevated due to high betweenness (0.4)
        // even though only 1 community is directly affected
        // betweenness_score * 0.5 = 0.40 * 3.0 clamped to 1.0 * 0.5 = 0.5
        // That alone pushes risk_score > 0.3 → at least "medium"
        let risk = json["risk_level"].as_str().unwrap();
        assert!(
            risk == "medium" || risk == "high",
            "Bridge with betweenness 0.4 should be at least medium risk, got: {}",
            risk
        );
    }

    #[tokio::test]
    async fn test_impact_leaf_file_low_risk() {
        // Test 2: Leaf file with low betweenness, 1 community → low risk
        let app = test_app_with_impact_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/impact?target=src/leaf.rs&project_slug=impact-proj",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["risk_level"], "low");
        assert!(json["betweenness_score"].as_f64().unwrap() < 0.05);
        let formula = json["risk_formula"].as_str().unwrap();
        assert!(formula.contains("low bridge"), "Formula: {}", formula);
    }

    #[tokio::test]
    async fn test_impact_no_gds_data_fallback() {
        // Test 3: File without GDS analytics → fallback to legacy threshold logic
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

        // No GDS data → betweenness_score and risk_formula should be absent
        assert!(
            json.get("betweenness_score").is_none() || json["betweenness_score"].is_null(),
            "betweenness_score should be absent without GDS data"
        );
        assert!(
            json.get("risk_formula").is_none() || json["risk_formula"].is_null(),
            "risk_formula should be absent without GDS data"
        );
        // affected_communities should be empty (no analytics)
        assert!(
            json.get("affected_communities").is_none()
                || json["affected_communities"]
                    .as_array()
                    .is_none_or(|c| c.is_empty()),
            "affected_communities should be empty without GDS data"
        );
        // Legacy risk level still works
        assert!(["low", "medium", "high"].contains(&json["risk_level"].as_str().unwrap()));
    }
}
