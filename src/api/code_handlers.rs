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
use crate::graph::algorithms::into_ranked;
use crate::graph::models::{FusionWeights, MultiSignalImpact, MultiSignalScore, RankedList};
use crate::neo4j::models::{ConnectedFileNode, DecisionNode};

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

/// Search response with both legacy hits and ranked view (Plan 10).
#[derive(Serialize)]
pub struct CodeSearchResult {
    /// Legacy: flat list of SearchHit (retro-compatible)
    pub hits:
        Vec<crate::meilisearch::indexes::SearchHit<crate::meilisearch::indexes::CodeDocument>>,
    /// Ranked view with margins and natural clusters.
    /// Uses CodeDocument directly (Meilisearch score as ranking score).
    pub ranked: RankedList<crate::meilisearch::indexes::CodeDocument>,
}

/// Build a CodeSearchResult from raw Meilisearch hits
fn build_search_result(
    hits: Vec<crate::meilisearch::indexes::SearchHit<crate::meilisearch::indexes::CodeDocument>>,
) -> CodeSearchResult {
    let scored: Vec<(crate::meilisearch::indexes::CodeDocument, f64)> =
        hits.iter().map(|h| (h.document.clone(), h.score)).collect();
    let total = scored.len();
    let ranked = into_ranked(scored, total);
    CodeSearchResult { hits, ranked }
}

/// Search code semantically across the codebase
///
/// Returns a `CodeSearchResult` with both legacy `hits` array and
/// `ranked` view with margins and clusters (Plan 10).
///
/// When `workspace_slug` is provided (and `project_slug` is not), searches all
/// projects in the workspace, merges results by score, and truncates to `limit`.
pub async fn search_code(
    State(state): State<OrchestratorState>,
    Query(params): Query<CodeSearchQuery>,
) -> Result<Json<CodeSearchResult>, AppError> {
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
        return Ok(Json(build_search_result(hits)));
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

        return Ok(Json(build_search_result(all_hits)));
    }

    // No filter — global search
    let hits = state
        .orchestrator
        .meili()
        .search_code_with_scores(&params.query, limit, params.language.as_deref(), None, None)
        .await?;

    Ok(Json(build_search_result(hits)))
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReference {
    pub file_path: String,
    pub line: u32,
    pub context: String,
    pub reference_type: String, // "call", "import", "type_usage"
}

/// Response for find_references with both legacy array and ranked view.
#[derive(Serialize)]
pub struct FindReferencesResult {
    /// Legacy: flat list of references (retro-compatible)
    pub references: Vec<SymbolReference>,
    /// Ranked view with margins and clusters (Plan 10).
    /// Calls scored 1.0, imports 0.7, type_usage 0.5.
    pub ranked: RankedList<SymbolReference>,
}

/// Find all references to a symbol across the codebase
pub async fn find_references(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindReferencesQuery>,
) -> Result<Json<FindReferencesResult>, AppError> {
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

    // Score references by type: calls are most important, then imports, then type_usage
    let scored: Vec<(SymbolReference, f64)> = references
        .iter()
        .map(|r| {
            let type_score = match r.reference_type.as_str() {
                "call" => 1.0,
                "import" => 0.7,
                "type_usage" => 0.5,
                _ => 0.3,
            };
            (r.clone(), type_score)
        })
        .collect();
    let total = scored.len();
    let ranked = into_ranked(scored, total);

    Ok(Json(FindReferencesResult { references, ranked }))
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
pub struct CallGraphEdge {
    pub name: String,
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub struct CallGraphNode {
    pub name: String,
    pub file_path: String,
    pub line: u32,
    pub callers: Vec<String>,
    pub callees: Vec<String>,
    /// Detailed caller info including confidence scores (when available)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub caller_details: Vec<CallGraphEdge>,
    /// Detailed callee info including confidence scores (when available)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub callee_details: Vec<CallGraphEdge>,
}

/// Get call graph for a function
pub async fn get_call_graph(
    State(state): State<OrchestratorState>,
    Query(query): Query<CallGraphQuery>,
) -> Result<Json<CallGraphNode>, AppError> {
    let depth = query.depth.unwrap_or(2).clamp(1, 20);
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
    let mut caller_details = vec![];
    let mut callee_details = vec![];

    if direction == "callers" || direction == "both" {
        callers = state
            .orchestrator
            .neo4j()
            .get_function_callers_by_name(&query.function, depth, project_id)
            .await?;
        // Also get direct callers with confidence (depth 1)
        if let Ok(details) = state
            .orchestrator
            .neo4j()
            .get_callers_with_confidence(&query.function, project_id)
            .await
        {
            caller_details = details
                .into_iter()
                .map(|(name, file, conf, reason)| CallGraphEdge {
                    name,
                    file_path: file,
                    confidence: Some(conf),
                    reason: Some(reason),
                })
                .collect();
        }
    }

    if direction == "callees" || direction == "both" {
        callees = state
            .orchestrator
            .neo4j()
            .get_function_callees_by_name(&query.function, depth, project_id)
            .await?;
        if let Ok(details) = state
            .orchestrator
            .neo4j()
            .get_callees_with_confidence(&query.function, project_id)
            .await
        {
            callee_details = details
                .into_iter()
                .map(|(name, file, conf, reason)| CallGraphEdge {
                    name,
                    file_path: file,
                    confidence: Some(conf),
                    reason: Some(reason),
                })
                .collect();
        }
    }

    Ok(Json(CallGraphNode {
        name: query.function,
        file_path: String::new(),
        line: 0,
        callers,
        callees,
        caller_details,
        callee_details,
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
    /// Analysis profile name or id (e.g. "default", "security", "refactoring")
    /// Used to weight edges differently in impact analysis.
    pub profile: Option<String>,
}

/// A single affected file with its path — used as the item type for `RankedList`.
/// The `#[serde(flatten)]` on `RankedResult<AffectedFile>` puts `path` at top level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedFile {
    pub path: String,
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
    /// Analysis profile used for risk weighting (None = default hardcoded weights)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_name: Option<String>,
    /// Community labels affected by this change (from graph analytics)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub affected_communities: Vec<String>,
    /// Betweenness centrality of the target node (bridge score)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub betweenness_score: Option<f64>,
    /// Human-readable explanation of the risk score computation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_formula: Option<String>,
    /// Architectural decisions that AFFECT the target or its impacted files
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub affecting_decisions: Vec<DecisionNode>,
    /// Ranked view of all affected files with margins and natural clusters (Plan 10).
    /// Direct files score 1.0, transitive-only files score 0.33.
    pub ranked_affected: RankedList<AffectedFile>,
    /// Pre-computed context cards for directly affected files (Plan 8).
    /// Each card contains PageRank, betweenness, DNA, WL hash, co-changers, etc.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_cards: Vec<crate::graph::models::ContextCard>,
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

    // Resolve analysis profile (by name or id, optional)
    let profile = if let Some(ref profile_ref) = query.profile {
        // Try to find by name among built-in profiles first
        let builtins = crate::graph::models::builtin_profiles();
        let found = builtins
            .into_iter()
            .find(|p| p.name == *profile_ref || p.id == *profile_ref);
        match found {
            Some(p) => Some(p),
            None => {
                // Try Neo4j lookup by id
                state
                    .orchestrator
                    .neo4j()
                    .get_analysis_profile(profile_ref)
                    .await
                    .unwrap_or(None)
            }
        }
    } else {
        None
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
    // Profile fusion weights adjust the relative importance of each factor:
    //   bridge     → betweenness weight
    //   co_change  → community spread weight
    //   structural → degree/caller weight
    let (w_betweenness, w_community, w_degree) = if let Some(ref p) = profile {
        let fw = &p.fusion_weights;
        let total = fw.bridge + fw.co_change + fw.structural;
        if total > 0.0 {
            (
                fw.bridge / total,
                fw.co_change / total,
                fw.structural / total,
            )
        } else {
            (0.5, 0.3, 0.2)
        }
    } else {
        (0.5, 0.3, 0.2)
    };

    let (risk_level, betweenness_score, risk_formula) = if let Some(ref analytics) = node_analytics
    {
        if let Some(betweenness) = analytics.betweenness {
            // Composite risk formula with profile-weighted coefficients:
            // betweenness_score = clamp(betweenness * 3.0, 0, 1)
            // community_spread = affected_communities / total (use 5 as reasonable estimate)
            // degree_score = clamp(caller_count / 20.0, 0, 1)
            // risk = betweenness_score * w_b + community_spread * w_c + degree_score * w_d
            let bs = (betweenness * 3.0).clamp(0.0, 1.0);
            let total_communities = 5.0_f64; // reasonable default
            let cs = (affected_communities.len() as f64 / total_communities).clamp(0.0, 1.0);
            let ds = (caller_count as f64 / 20.0).clamp(0.0, 1.0);
            let risk_score = bs * w_betweenness + cs * w_community + ds * w_degree;

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

    // Fetch architectural decisions that AFFECT the target and its direct dependencies.
    // Uses the reverse AFFECTS lookup: (Decision)-[:AFFECTS]->(File/Function).
    // Best-effort: returns empty vec if no AFFECTS relations exist yet.
    let affecting_decisions = {
        let entity_type_label = match target_type {
            "function" => "Function",
            _ => "File",
        };
        // Get decisions affecting the target itself
        let mut decisions = state
            .orchestrator
            .neo4j()
            .get_decisions_affecting(entity_type_label, &target, None)
            .await
            .unwrap_or_default();

        // Also check decisions affecting directly affected files (deduplicated)
        for affected_file in directly_affected.iter().take(5) {
            let file_decisions = state
                .orchestrator
                .neo4j()
                .get_decisions_affecting("File", affected_file, None)
                .await
                .unwrap_or_default();
            for d in file_decisions {
                if !decisions.iter().any(|existing| existing.id == d.id) {
                    decisions.push(d);
                }
            }
        }
        decisions
    };

    // Batch-read context cards for directly affected files (best-effort)
    let context_cards = if let Some(ref pid) = project_id {
        state
            .orchestrator
            .neo4j()
            .get_context_cards_batch(&directly_affected, &pid.to_string())
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // Build ranked view: direct files score 1.0, transitive-only score 0.33
    let ranked_affected = {
        let mut scored: Vec<(AffectedFile, f64)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for path in &directly_affected {
            if seen.insert(path.clone()) {
                scored.push((AffectedFile { path: path.clone() }, 1.0));
            }
        }
        for path in &transitively_affected {
            if seen.insert(path.clone()) {
                scored.push((AffectedFile { path: path.clone() }, 0.33));
            }
        }
        let total = scored.len();
        into_ranked(scored, total)
    };

    Ok(Json(ImpactAnalysis {
        target,
        directly_affected,
        transitively_affected,
        test_files_affected: test_files,
        caller_count,
        risk_level,
        suggestion,
        profile_name: profile.as_ref().map(|p| p.name.clone()),
        affected_communities,
        betweenness_score,
        risk_formula,
        affecting_decisions,
        ranked_affected,
        context_cards,
    }))
}

// ============================================================================
// Multi-signal Impact Fusion (Plan 4)
// ============================================================================

/// Query parameters for analyze_impact_v2 (same as ImpactQuery but reused)
#[derive(Deserialize)]
pub struct MultiImpactQuery {
    /// File path to analyze
    pub target: String,
    /// Filter by project slug (required for multi-signal)
    pub project_slug: String,
    /// Analysis profile name or id (defaults to "default")
    pub profile: Option<String>,
}

/// Multi-signal impact analysis: 5 signals fused with configurable weights.
/// All 5 signals are queried in parallel via tokio::join!
pub async fn analyze_impact_v2(
    State(state): State<OrchestratorState>,
    Query(query): Query<MultiImpactQuery>,
) -> Result<Json<MultiSignalImpact>, AppError> {
    let start = std::time::Instant::now();
    let neo4j = state.orchestrator.neo4j();

    // Resolve project
    let project = neo4j
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;
    let project_id = project.id;
    let project_id_str = project_id.to_string();

    // Resolve target path (relative → absolute)
    let target = if !query.target.starts_with('/') {
        let expanded = crate::expand_tilde(&project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &query.target)
    } else {
        query.target.clone()
    };

    // Resolve analysis profile (built-in first, then Neo4j)
    let profile = if let Some(ref profile_ref) = query.profile {
        let builtins = crate::graph::models::builtin_profiles();
        // TODO: fallback to async Neo4j profile lookup when not found in builtins
        builtins
            .into_iter()
            .find(|p| p.name == *profile_ref || p.id == *profile_ref)
    } else {
        None
    };

    let weights = profile
        .as_ref()
        .map(|p| p.fusion_weights.clone())
        .unwrap_or_else(|| FusionWeights {
            structural: 0.35,
            co_change: 0.25,
            knowledge: 0.15,
            pagerank: 0.10,
            bridge: 0.15,
        });
    let profile_name = profile
        .as_ref()
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "default".to_string());

    // ===== 5 signals in parallel via tokio::join! =====
    let (structural_res, co_change_res, knowledge_res, pagerank_res, bridge_res) = tokio::join!(
        // Signal 1: Structural impact (IMPORTS + CALLS traversal)
        neo4j.find_impacted_files(&target, 3, Some(project_id)),
        // Signal 2: Co-change (temporal coupling)
        neo4j.get_file_co_changers(&target, 1, 20),
        // Signal 3: Knowledge density (notes + decisions linked)
        neo4j.get_knowledge_density(&target, &project_id_str),
        // Signal 4: PageRank (structural importance)
        neo4j.get_node_pagerank(&target, &project_id_str),
        // Signal 5: Bridge proximity (shortest path to co-changers)
        neo4j.get_bridge_proximity(&target, &project_id_str),
    );

    // Collect results (best-effort: log errors, use defaults)
    let structural = structural_res.unwrap_or_default();
    let co_changers = co_change_res.unwrap_or_default();
    let target_knowledge = knowledge_res.unwrap_or(0.0);
    let target_pagerank = pagerank_res.unwrap_or(0.0);
    let bridge_scores = bridge_res.unwrap_or_default();

    // ===== Merge into HashMap<path, MultiSignalScore> =====
    let mut scores: std::collections::HashMap<String, MultiSignalScore> =
        std::collections::HashMap::new();

    // Signal 1: Structural (1.0 for direct, 0.33 for transitive)
    for (i, path) in structural.iter().enumerate() {
        let entry = scores
            .entry(path.clone())
            .or_insert_with(|| MultiSignalScore {
                path: path.clone(),
                ..Default::default()
            });
        // First file is direct (1.0), rest attenuated by position
        entry.structural_score = if i < 5 { 1.0 } else { 0.33 };
        entry.signals.push("structural".to_string());
    }

    // Signal 2: Co-change (normalize by max count)
    let max_co_change = co_changers
        .iter()
        .map(|c| c.count)
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    for co in &co_changers {
        let entry = scores
            .entry(co.path.clone())
            .or_insert_with(|| MultiSignalScore {
                path: co.path.clone(),
                ..Default::default()
            });
        entry.co_change_score = co.count as f64 / max_co_change;
        entry.signals.push("co_change".to_string());
    }

    // Signal 3: Knowledge density (target score propagated to all entries)
    // Each file gets the target's knowledge density as a proxy
    // (querying per-file would be too slow for the 500ms budget)
    if target_knowledge > 0.0 {
        for score in scores.values_mut() {
            score.knowledge_score = target_knowledge;
            if !score.signals.contains(&"knowledge".to_string()) {
                score.signals.push("knowledge".to_string());
            }
        }
    }

    // Signal 4: PageRank (target score as global importance proxy)
    if target_pagerank > 0.0 {
        for score in scores.values_mut() {
            score.pagerank_score = target_pagerank;
            if !score.signals.contains(&"pagerank".to_string()) {
                score.signals.push("pagerank".to_string());
            }
        }
    }

    // Signal 5: Bridge proximity (per-file score from co-changer distance)
    for (path, proximity) in &bridge_scores {
        let entry = scores
            .entry(path.clone())
            .or_insert_with(|| MultiSignalScore {
                path: path.clone(),
                ..Default::default()
            });
        entry.bridge_score = *proximity;
        entry.signals.push("bridge".to_string());
    }

    // ===== Compute combined score with fusion weights =====
    for score in scores.values_mut() {
        score.combined_score = weights.structural * score.structural_score
            + weights.co_change * score.co_change_score
            + weights.knowledge * score.knowledge_score
            + weights.pagerank * score.pagerank_score
            + weights.bridge * score.bridge_score;
    }

    // ===== Build ranked list =====
    let mut scored_items: Vec<(MultiSignalScore, f64)> = scores
        .into_values()
        .map(|s| {
            let combined = s.combined_score;
            (s, combined)
        })
        .collect();
    scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total = scored_items.len();
    let ranked = into_ranked(scored_items, total);

    let timing_ms = start.elapsed().as_millis() as u64;

    Ok(Json(MultiSignalImpact {
        target,
        profile_used: profile_name,
        weights,
        ranked,
        timing_ms,
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
    let relations: Vec<serde_json::Value> = detail
        .relations
        .iter()
        .map(|r| {
            serde_json::json!({
                "source_type": r.source_type,
                "source_id": r.source_id,
                "target_type": r.target_type,
                "target_id": r.target_id,
                "relation_type": r.relation_type,
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
        "relations": relations,
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
    // Resolve the feature graph's project_id for proper scoping
    let project_id = state
        .orchestrator
        .neo4j()
        .get_feature_graph_detail(id)
        .await?
        .map(|detail| detail.graph.project_id);

    state
        .orchestrator
        .neo4j()
        .add_entity_to_feature_graph(
            id,
            &body.entity_type,
            &body.entity_id,
            body.role.as_deref(),
            project_id,
        )
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
    let relations: Vec<serde_json::Value> = detail
        .relations
        .iter()
        .map(|r| {
            serde_json::json!({
                "source_type": r.source_type,
                "source_id": r.source_id,
                "target_type": r.target_type,
                "target_id": r.target_id,
                "relation_type": r.relation_type,
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
        "relations": relations,
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
                "unique_fingerprints": c.unique_fingerprints,
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

    // Best-effort: fetch hotspots, knowledge gaps, and risk summary.
    // Returns empty arrays / null if properties have not been computed yet.
    let hotspots = state
        .orchestrator
        .neo4j()
        .get_top_hotspots(project.id, 5)
        .await
        .unwrap_or_default();

    let knowledge_gaps = state
        .orchestrator
        .neo4j()
        .get_top_knowledge_gaps(project.id, 5)
        .await
        .unwrap_or_default();

    let risk_assessment = state
        .orchestrator
        .neo4j()
        .get_risk_summary(project.id)
        .await
        .unwrap_or(serde_json::json!(null));

    // Neural metrics (SYNAPSE layer health)
    let neural_metrics = match state
        .orchestrator
        .neo4j()
        .get_neural_metrics(project.id)
        .await
    {
        Ok(nm) => serde_json::json!({
            "active_synapses": nm.active_synapses,
            "avg_energy": nm.avg_energy,
            "weak_synapses_ratio": nm.weak_synapses_ratio,
            "dead_notes_count": nm.dead_notes_count,
        }),
        Err(_) => serde_json::json!(null),
    };

    // Multi-signal impact score (average combined score of top-10 files by PageRank)
    let avg_impact_score = state
        .orchestrator
        .neo4j()
        .get_avg_multi_signal_score(project.id)
        .await
        .unwrap_or(0.0);

    // Topology violations (best-effort — returns null if no rules defined)
    let topology_violations = match state
        .orchestrator
        .neo4j()
        .check_topology_rules(&project.id.to_string())
        .await
    {
        Ok(violations) => {
            let errors = violations
                .iter()
                .filter(|v| v.severity == crate::graph::models::TopologySeverity::Error)
                .count();
            let warnings = violations.len() - errors;
            serde_json::json!({
                "errors": errors,
                "warnings": warnings,
                "total": violations.len(),
            })
        }
        Err(_) => serde_json::json!(null),
    };

    // Homeostasis pain score (best-effort)
    let homeostasis = state
        .orchestrator
        .neo4j()
        .compute_homeostasis(project.id, None)
        .await
        .ok();

    let homeostasis_json = homeostasis.as_ref().map(|h| {
        serde_json::json!({
            "pain_score": h.pain_score,
            "ratios": h.ratios.iter().map(|r| serde_json::json!({
                "name": r.name,
                "value": r.value,
                "severity": r.severity,
                "distance": r.distance_to_equilibrium,
            })).collect::<Vec<_>>(),
            "recommendations": h.recommendations,
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
        "hotspots": hotspots,
        "knowledge_gaps": knowledge_gaps,
        "risk_assessment": risk_assessment,
        "neural_metrics": neural_metrics,
        "avg_impact_score": avg_impact_score,
        "topology_violations": topology_violations,
        "homeostasis": homeostasis_json,
    })))
}

// ============================================================================
// Node Importance (GDS Analytics)
// ============================================================================

#[derive(Deserialize)]
pub struct NodeImportanceQuery {
    pub project_slug: String,
    pub node_path: String,
    pub node_type: Option<String>,
}

/// GET /api/code/node-importance — Get GDS metrics for a specific node
pub async fn get_node_importance(
    State(state): State<OrchestratorState>,
    Query(params): Query<NodeImportanceQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let node_type = params.node_type.as_deref().unwrap_or("file");

    let metrics = state
        .orchestrator
        .neo4j()
        .get_node_gds_metrics(&params.node_path, node_type, project.id)
        .await
        .map_err(AppError::Internal)?;

    let metrics = match metrics {
        Some(m) => m,
        None => {
            return Err(AppError::NotFound(format!(
                "Node not found: '{}' (type: {})",
                params.node_path, node_type
            )));
        }
    };

    if metrics.pagerank.is_none() {
        return Ok(Json(serde_json::json!({
            "node": params.node_path,
            "node_type": node_type,
            "message": "No structural analytics available for this node. Run sync_project first, then wait for analytics computation.",
            "metrics": {
                "in_degree": metrics.in_degree,
                "out_degree": metrics.out_degree,
            }
        })));
    }

    let percentiles = state
        .orchestrator
        .neo4j()
        .get_project_percentiles(project.id)
        .await
        .map_err(AppError::Internal)?;

    // Calculate risk level based on PageRank + betweenness
    let pagerank = metrics.pagerank.unwrap_or(0.0);
    let betweenness = metrics.betweenness.unwrap_or(0.0);

    let risk_level =
        if pagerank > percentiles.pagerank_p95 && betweenness > percentiles.betweenness_p95 {
            "critical"
        } else if pagerank > percentiles.pagerank_p95 || betweenness > percentiles.betweenness_p95 {
            "high"
        } else if pagerank > percentiles.pagerank_p80 || betweenness > percentiles.betweenness_p80 {
            "medium"
        } else {
            "low"
        };

    let summary = match risk_level {
        "critical" => format!("{} has very high PageRank and betweenness — modifying it has significant regression risk", params.node_path),
        "high" => format!("{} has high centrality — changes may have wide impact", params.node_path),
        "medium" => format!("{} has moderate importance — standard review recommended", params.node_path),
        _ => format!("{} has low centrality — changes are relatively safe", params.node_path),
    };

    Ok(Json(serde_json::json!({
        "node": params.node_path,
        "node_type": node_type,
        "risk_level": risk_level,
        "summary": summary,
        "metrics": {
            "pagerank": pagerank,
            "betweenness": betweenness,
            "clustering_coefficient": metrics.clustering_coefficient,
            "community_id": metrics.community_id,
            "in_degree": metrics.in_degree,
            "out_degree": metrics.out_degree,
        },
        "fabric_metrics": {
            "fabric_pagerank": metrics.fabric_pagerank,
            "fabric_betweenness": metrics.fabric_betweenness,
            "fabric_community_id": metrics.fabric_community_id,
            "fabric_community_label": metrics.fabric_community_label,
        },
        "percentiles": {
            "pagerank_p80": percentiles.pagerank_p80,
            "pagerank_p95": percentiles.pagerank_p95,
            "betweenness_p80": percentiles.betweenness_p80,
            "betweenness_p95": percentiles.betweenness_p95,
        }
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

// ============================================================================
// T5.5 — Change Hotspots (Churn Score)
// ============================================================================

#[derive(Deserialize)]
pub struct HotspotsQuery {
    pub project_slug: String,
    pub limit: Option<usize>,
}

/// GET /api/code/hotspots — Get files sorted by churn score (most frequently changed first)
pub async fn get_change_hotspots(
    State(state): State<OrchestratorState>,
    Query(params): Query<HotspotsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let limit = params.limit.unwrap_or(20);
    let scores = state
        .orchestrator
        .neo4j()
        .compute_churn_scores(project.id)
        .await?;

    let limited: Vec<&crate::neo4j::models::FileChurnScore> = scores.iter().take(limit).collect();

    Ok(Json(serde_json::json!({
        "hotspots": limited,
        "total_files": scores.len(),
        "limit": limit,
    })))
}

// ============================================================================
// T5.6 — Knowledge Gaps (Knowledge Density)
// ============================================================================

#[derive(Deserialize)]
pub struct KnowledgeGapsQuery {
    pub project_slug: String,
    pub limit: Option<usize>,
}

/// GET /api/code/knowledge-gaps — Get files sorted by knowledge density ASC (least documented first)
pub async fn get_knowledge_gaps(
    State(state): State<OrchestratorState>,
    Query(params): Query<KnowledgeGapsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let limit = params.limit.unwrap_or(20);
    let mut scores = state
        .orchestrator
        .neo4j()
        .compute_knowledge_density(project.id)
        .await?;

    // Sort by knowledge_density ASC (least documented first)
    scores.sort_by(|a, b| {
        a.knowledge_density
            .partial_cmp(&b.knowledge_density)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let limited: Vec<&crate::neo4j::models::FileKnowledgeDensity> =
        scores.iter().take(limit).collect();

    Ok(Json(serde_json::json!({
        "knowledge_gaps": limited,
        "total_files": scores.len(),
        "limit": limit,
    })))
}

// ============================================================================
// T5.7 — Risk Assessment (Composite Risk Score)
// ============================================================================

#[derive(Deserialize)]
pub struct RiskAssessmentQuery {
    pub project_slug: String,
    pub limit: Option<usize>,
}

/// GET /api/code/risk-assessment — Get files sorted by composite risk score DESC
pub async fn get_risk_assessment(
    State(state): State<OrchestratorState>,
    Query(params): Query<RiskAssessmentQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let limit = params.limit.unwrap_or(20);
    let scores = state
        .orchestrator
        .neo4j()
        .compute_risk_scores(project.id)
        .await?;

    let limited: Vec<&crate::neo4j::models::FileRiskScore> = scores.iter().take(limit).collect();

    // Compute summary stats
    let total = scores.len();
    let critical = scores.iter().filter(|s| s.risk_level == "critical").count();
    let high = scores.iter().filter(|s| s.risk_level == "high").count();
    let medium = scores.iter().filter(|s| s.risk_level == "medium").count();
    let low = scores.iter().filter(|s| s.risk_level == "low").count();
    let avg_risk = if total > 0 {
        scores.iter().map(|s| s.risk_score).sum::<f64>() / total as f64
    } else {
        0.0
    };

    Ok(Json(serde_json::json!({
        "risk_files": limited,
        "total_files": total,
        "limit": limit,
        "summary": {
            "avg_risk_score": avg_risk,
            "critical_count": critical,
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
        }
    })))
}

// ============================================================================
// Homeostasis — Bio-inspired auto-regulation
// ============================================================================

#[derive(Deserialize)]
pub struct HomeostasisQuery {
    pub project_slug: String,
}

/// GET /api/code/homeostasis — Get homeostatic equilibrium report for the knowledge graph
pub async fn get_homeostasis(
    State(state): State<OrchestratorState>,
    Query(params): Query<HomeostasisQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let report = state
        .orchestrator
        .neo4j()
        .compute_homeostasis(project.id, None)
        .await?;

    Ok(Json(serde_json::to_value(report).unwrap_or_default()))
}

// ============================================================================
// Identity Manifold — Structural Drift
// ============================================================================

/// Query params for GET /api/code/structural-drift
#[derive(Debug, Deserialize)]
pub struct StructuralDriftQuery {
    pub project_slug: String,
    /// Warning threshold (default: 1.5)
    pub warning_threshold: Option<f64>,
    /// Critical threshold (default: 3.0)
    pub critical_threshold: Option<f64>,
}

/// GET /api/code/structural-drift
///
/// Compute structural drift for all files in a project.
/// Returns community centroids and per-file drift from their community identity.
/// Biomimicry: maps to Elun's HypersphereIdentity distance_squared_f64.
pub async fn get_structural_drift(
    State(state): State<OrchestratorState>,
    Query(params): Query<StructuralDriftQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let report = state
        .orchestrator
        .neo4j()
        .compute_structural_drift(
            project.id,
            params.warning_threshold,
            params.critical_threshold,
        )
        .await?;

    Ok(Json(serde_json::to_value(report).unwrap_or_default()))
}

// ============================================================================
// Process Detection
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ProjectSlugBody {
    pub project_slug: String,
}

/// POST /api/code/processes/detect
///
/// Detect business processes by scoring entry points, BFS traversal through
/// the CALLS graph, deduplication, and classification.
pub async fn detect_processes(
    State(state): State<OrchestratorState>,
    Json(body): Json<ProjectSlugBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let processes = state
        .orchestrator
        .analytics()
        .detect_processes(project.id)
        .await?;

    let processes_json: Vec<serde_json::Value> = processes
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "label": p.label,
                "process_type": p.process_type.to_string(),
                "step_count": p.steps.len(),
                "entry_point": p.entry_point_id,
                "terminal": p.terminal_id,
                "steps": p.steps,
                "communities": p.communities.iter().collect::<Vec<_>>(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "processes": processes_json,
        "total": processes.len(),
    })))
}

// ============================================================================
// Heritage navigation handlers
// ============================================================================

#[derive(Deserialize)]
pub struct ClassHierarchyQuery {
    pub type_name: String,
    pub max_depth: Option<u32>,
}

/// GET /api/code/class-hierarchy
///
/// Get the full class hierarchy (parents + children) for a type.
pub async fn get_class_hierarchy(
    State(state): State<OrchestratorState>,
    Query(params): Query<ClassHierarchyQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let max_depth = params.max_depth.unwrap_or(10).clamp(1, 20);
    let hierarchy = state
        .orchestrator
        .neo4j()
        .get_class_hierarchy(&params.type_name, max_depth)
        .await?;

    Ok(Json(hierarchy))
}

#[derive(Deserialize)]
pub struct SubclassesQuery {
    pub class_name: String,
}

/// GET /api/code/subclasses
///
/// Find all subclasses (direct + transitive) of a given class.
pub async fn find_subclasses(
    State(state): State<OrchestratorState>,
    Query(params): Query<SubclassesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let subclasses = state
        .orchestrator
        .neo4j()
        .find_subclasses(&params.class_name)
        .await?;

    Ok(Json(serde_json::json!({
        "class_name": params.class_name,
        "subclasses": subclasses,
        "total": subclasses.len(),
    })))
}

#[derive(Deserialize)]
pub struct InterfaceImplementorsQuery {
    pub interface_name: String,
}

/// GET /api/code/interface-implementors
///
/// Find all types that implement a given interface.
pub async fn find_interface_implementors(
    State(state): State<OrchestratorState>,
    Query(params): Query<InterfaceImplementorsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let implementors = state
        .orchestrator
        .neo4j()
        .find_interface_implementors(&params.interface_name)
        .await?;

    Ok(Json(serde_json::json!({
        "interface_name": params.interface_name,
        "implementors": implementors,
        "total": implementors.len(),
    })))
}

// ============================================================================
// Process navigation handlers
// ============================================================================

#[derive(Deserialize)]
pub struct ListProcessesQuery {
    pub project_slug: String,
}

/// GET /api/code/processes
///
/// List all detected processes for a project.
pub async fn list_processes(
    State(state): State<OrchestratorState>,
    Query(params): Query<ListProcessesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let processes = state
        .orchestrator
        .neo4j()
        .list_processes(project.id)
        .await?;

    Ok(Json(serde_json::json!({
        "processes": processes,
        "total": processes.len(),
    })))
}

#[derive(Deserialize)]
pub struct GetProcessQuery {
    pub process_id: String,
}

/// GET /api/code/processes/detail
///
/// Get details of a specific process including ordered steps.
pub async fn get_process_detail(
    State(state): State<OrchestratorState>,
    Query(params): Query<GetProcessQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let process = state
        .orchestrator
        .neo4j()
        .get_process_detail(&params.process_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Process '{}' not found", params.process_id)))?;

    Ok(Json(process))
}

#[derive(Deserialize)]
pub struct EntryPointsQuery {
    pub project_slug: String,
    pub limit: Option<usize>,
}

/// GET /api/code/entry-points
///
/// Get scored entry points for a project.
pub async fn get_entry_points(
    State(state): State<OrchestratorState>,
    Query(params): Query<EntryPointsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", params.project_slug))
        })?;

    let limit = params.limit.unwrap_or(50);
    let entry_points = state
        .orchestrator
        .neo4j()
        .get_entry_points(project.id, limit)
        .await?;

    Ok(Json(serde_json::json!({
        "entry_points": entry_points,
        "total": entry_points.len(),
    })))
}

// ============================================================================
// Community enrichment handler
// ============================================================================

/// POST /api/code/communities/enrich
///
/// Trigger LLM enrichment of community labels.
pub async fn enrich_communities(
    State(state): State<OrchestratorState>,
    Json(body): Json<ProjectSlugBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    // Run file graph analysis which includes enrichment
    let analytics = state
        .orchestrator
        .analytics()
        .analyze_file_graph(project.id)
        .await?;

    let communities_json: Vec<serde_json::Value> = analytics
        .communities
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "label": c.label,
                "size": c.size,
                "cohesion": c.cohesion,
                "enriched_by": c.enriched_by,
                "members": c.members,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "communities": communities_json,
        "total": communities_json.len(),
    })))
}

// ============================================================================
// Bridge Subgraph (Plan 1 — GraIL)
// ============================================================================

#[derive(Deserialize)]
pub struct BridgeQuery {
    /// Source node path (file or function)
    pub source: String,
    /// Target node path (file or function)
    pub target: String,
    /// Project slug (required for scoping)
    pub project_slug: String,
    /// Max BFS hops (default 3, clamped 1-5)
    pub max_hops: Option<u32>,
    /// Number of bottleneck nodes to return (default 3)
    pub top_bottlenecks: Option<usize>,
}

/// GET /api/code/bridge — Extract the GraIL-style bridge subgraph between two nodes.
///
/// Returns an enriched `BridgeSubgraph` with double-radius labeling,
/// density, and bottleneck detection (Brandes' betweenness centrality).
pub async fn get_bridge(
    State(state): State<OrchestratorState>,
    Query(params): Query<BridgeQuery>,
) -> Result<Json<crate::graph::models::BridgeSubgraph>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    // Resolve project
    let project = neo4j
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", params.project_slug)))?;
    let project_id_str = project.id.to_string();

    // Resolve paths (relative → absolute)
    let root = crate::expand_tilde(&project.root_path);
    let root = root.trim_end_matches('/');
    let source = if !params.source.starts_with('/') {
        format!("{}/{}", root, &params.source)
    } else {
        params.source.clone()
    };
    let target = if !params.target.starts_with('/') {
        format!("{}/{}", root, &params.target)
    } else {
        params.target.clone()
    };

    let max_hops = params.max_hops.unwrap_or(3).clamp(1, 5);
    let top_n = params.top_bottlenecks.unwrap_or(3);

    // Default relation types for bridge extraction
    let relation_types: Vec<String> = vec![
        "IMPORTS".to_string(),
        "CALLS".to_string(),
        "CO_CHANGED".to_string(),
        "EXTENDS".to_string(),
        "IMPLEMENTS".to_string(),
    ];

    // Phase 1: Extract raw bridge subgraph from Neo4j
    let (raw_nodes, raw_edges) = neo4j
        .find_bridge_subgraph(&source, &target, max_hops, &relation_types, &project_id_str)
        .await?;

    if raw_nodes.is_empty() {
        return Err(AppError::NotFound(format!(
            "No bridge subgraph found between '{}' and '{}' within {} hops",
            source, target, max_hops
        )));
    }

    // Phase 2: Enrich with algorithms (double-radius labeling + bottleneck detection)
    let node_paths: Vec<String> = raw_nodes.iter().map(|n| n.path.clone()).collect();
    let edge_tuples: Vec<(String, String)> = raw_edges
        .iter()
        .map(|e| (e.from_path.clone(), e.to_path.clone()))
        .collect();

    let labels =
        crate::graph::algorithms::double_radius_label(&node_paths, &edge_tuples, &source, &target);
    let bottleneck_nodes = crate::graph::algorithms::find_bottleneck_nodes(
        &node_paths,
        &edge_tuples,
        &source,
        &target,
        top_n,
    );
    let density =
        crate::graph::algorithms::compute_bridge_density(raw_nodes.len(), raw_edges.len());

    // Phase 3: Assemble enriched BridgeSubgraph
    let nodes: Vec<crate::graph::models::BridgeNode> = raw_nodes
        .iter()
        .map(|n| {
            let (d_s, d_t) = labels.get(&n.path).copied().unwrap_or((u32::MAX, u32::MAX));
            crate::graph::models::BridgeNode {
                path: n.path.clone(),
                node_type: n.node_type.clone(),
                distance_to_source: d_s,
                distance_to_target: d_t,
            }
        })
        .collect();

    let edges: Vec<crate::graph::models::BridgeEdge> = raw_edges
        .iter()
        .map(|e| crate::graph::models::BridgeEdge {
            from_path: e.from_path.clone(),
            to_path: e.to_path.clone(),
            rel_type: e.rel_type.clone(),
        })
        .collect();

    let result = crate::graph::models::BridgeSubgraph {
        source,
        target,
        nodes,
        edges,
        density,
        bottleneck_nodes,
    };

    // Phase 4: Hebbian synapse reinforcement (fire-and-forget)
    // "Notes that bridge together, wire together"
    // Collect notes linked to bridge nodes and reinforce synapses between them.
    let bridge_node_paths: Vec<String> = node_paths;
    let neo4j_bg = state.orchestrator.neo4j_arc();
    tokio::spawn(async move {
        let mut all_note_ids: Vec<uuid::Uuid> = Vec::new();
        for path in &bridge_node_paths {
            if let Ok(notes) = neo4j_bg
                .get_notes_for_entity(&crate::notes::EntityType::File, path)
                .await
            {
                for note in &notes {
                    all_note_ids.push(note.id);
                }
            }
        }
        all_note_ids.sort();
        all_note_ids.dedup();
        if all_note_ids.len() >= 2 {
            match neo4j_bg.reinforce_synapses(&all_note_ids, 0.05).await {
                Ok(count) => {
                    tracing::debug!(
                        reinforced = count,
                        bridge_nodes = bridge_node_paths.len(),
                        notes = all_note_ids.len(),
                        "Bridge subgraph: Hebbian synapse reinforcement completed"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Bridge subgraph: Hebbian synapse reinforcement failed"
                    );
                }
            }
        }
    });

    Ok(Json(result))
}

// ============================================================================
// Topological Firewall (Plan 3 — GraIL)
// ============================================================================

#[derive(Deserialize)]
pub struct TopologyCheckQuery {
    /// Project slug (required for scoping)
    pub project_slug: String,
}

/// GET /api/code/topology/check — Check all topology rules for violations.
///
/// Returns a `TopologyCheckResult` with all violations found, sorted by
/// violation_score descending (most dangerous first).
pub async fn check_topology(
    State(state): State<OrchestratorState>,
    Query(params): Query<TopologyCheckQuery>,
) -> Result<Json<crate::graph::models::TopologyCheckResult>, AppError> {
    let start = std::time::Instant::now();
    let neo4j = state.orchestrator.neo4j();

    // Resolve project
    let project = neo4j
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", params.project_slug)))?;
    let project_id_str = project.id.to_string();

    // Count rules
    let rules = neo4j.list_topology_rules(&project_id_str).await?;
    let rules_checked = rules.len();

    // Check all rules
    let violations = neo4j.check_topology_rules(&project_id_str).await?;

    let error_count = violations
        .iter()
        .filter(|v| v.severity == crate::graph::models::TopologySeverity::Error)
        .count();
    let warning_count = violations
        .iter()
        .filter(|v| v.severity == crate::graph::models::TopologySeverity::Warning)
        .count();

    let timing_ms = start.elapsed().as_millis() as u64;

    Ok(Json(crate::graph::models::TopologyCheckResult {
        project_id: project_id_str,
        rules_checked,
        violations,
        error_count,
        warning_count,
        timing_ms,
    }))
}

#[derive(Deserialize)]
pub struct TopologyRulesQuery {
    /// Project slug (required for scoping)
    pub project_slug: String,
}

/// GET /api/code/topology/rules — List all topology rules for a project.
pub async fn list_topology_rules(
    State(state): State<OrchestratorState>,
    Query(params): Query<TopologyRulesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let project = neo4j
        .get_project_by_slug(&params.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", params.project_slug)))?;

    let rules = neo4j.list_topology_rules(&project.id.to_string()).await?;

    Ok(Json(serde_json::json!({
        "rules": rules,
        "total": rules.len(),
    })))
}

#[derive(Deserialize)]
pub struct CreateTopologyRuleBody {
    /// Project slug
    pub project_slug: String,
    /// Rule type: must_not_import, must_not_call, max_distance, max_fan_out, no_circular
    pub rule_type: String,
    /// Glob pattern for source files (e.g. "src/neo4j/**")
    pub source_pattern: String,
    /// Glob pattern for target files (e.g. "src/api/**") — optional for MaxFanOut, NoCircular
    pub target_pattern: Option<String>,
    /// Numeric threshold (for MaxDistance, MaxFanOut)
    pub threshold: Option<u32>,
    /// Severity: "error" or "warning" (default: "error")
    pub severity: Option<String>,
    /// Human-readable description
    pub description: String,
}

/// POST /api/code/topology/rules — Create a new topology rule.
pub async fn create_topology_rule(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateTopologyRuleBody>,
) -> Result<Json<crate::graph::models::TopologyRule>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let project = neo4j
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", body.project_slug)))?;

    let rule_type = crate::graph::models::TopologyRuleType::from_str_loose(&body.rule_type)
        .ok_or_else(|| {
            AppError::BadRequest(format!(
                "Invalid rule_type: {}. Valid: must_not_import, must_not_call, max_distance, max_fan_out, no_circular",
                body.rule_type
            ))
        })?;

    let severity = body
        .severity
        .as_deref()
        .map(crate::graph::models::TopologySeverity::from_str_loose)
        .unwrap_or(Some(crate::graph::models::TopologySeverity::Error))
        .ok_or_else(|| {
            AppError::BadRequest("Invalid severity. Valid: error, warning".to_string())
        })?;

    let rule = crate::graph::models::TopologyRule {
        id: uuid::Uuid::new_v4().to_string(),
        project_id: project.id.to_string(),
        rule_type,
        source_pattern: body.source_pattern,
        target_pattern: body.target_pattern,
        threshold: body.threshold,
        severity,
        description: body.description,
    };

    neo4j.create_topology_rule(&rule).await?;

    Ok(Json(rule))
}

/// DELETE /api/code/topology/rules/:rule_id — Delete a topology rule.
pub async fn delete_topology_rule(
    State(state): State<OrchestratorState>,
    Path(rule_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    neo4j.delete_topology_rule(&rule_id).await?;
    Ok(Json(serde_json::json!({"deleted": true})))
}

// ============================================================================
// Topology: Single-file check (real-time pre-write validation)
// ============================================================================

/// Request body for checking a single file's imports against topology rules.
#[derive(Deserialize)]
pub struct CheckFileTopologyBody {
    pub project_slug: String,
    pub file_path: String,
    pub new_imports: Vec<String>,
}

/// POST /api/code/topology/check-file
///
/// Real-time pre-write validation: checks if the given `new_imports` for
/// `file_path` would violate any MustNotImport/MustNotCall topology rules.
/// Designed for <50ms response time (in-memory regex matching after 1 Neo4j query).
pub async fn check_file_topology(
    State(state): State<OrchestratorState>,
    Json(body): Json<CheckFileTopologyBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let violations = state
        .orchestrator
        .neo4j()
        .check_file_topology(&project.id.to_string(), &body.file_path, &body.new_imports)
        .await?;

    Ok(Json(serde_json::json!({
        "file_path": body.file_path,
        "violations": violations,
        "violation_count": violations.len(),
        "has_violations": !violations.is_empty(),
    })))
}

// ============================================================================
// Structural DNA: Profile & Twins
// ============================================================================

/// Request body for structural DNA endpoints.
#[derive(Deserialize)]
pub struct StructuralDnaBody {
    pub project_slug: String,
    pub file_path: String,
    /// Max results for find_structural_twins (default 10)
    pub top_n: Option<usize>,
}

/// POST /api/code/structural-profile
///
/// Returns the structural DNA vector for a single file within a project.
/// The DNA is a K-dimensional distance vector from the file node to K anchor
/// nodes (highest PageRank), normalized to [0,1]. Requires prior analytics run.
pub async fn get_structural_profile(
    State(state): State<OrchestratorState>,
    Json(body): Json<StructuralDnaBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    // Resolve relative path to absolute
    let resolved_path = if !body.file_path.starts_with('/') {
        let expanded = crate::expand_tilde(&project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &body.file_path)
    } else {
        body.file_path.clone()
    };

    let all_dna = state
        .orchestrator
        .neo4j()
        .get_project_structural_dna(&project.id.to_string())
        .await?;

    // Find the target file's DNA
    let target_dna = all_dna
        .iter()
        .find(|(path, _)| path == &resolved_path)
        .map(|(_, dna)| dna.clone());

    match target_dna {
        Some(dna) => Ok(Json(serde_json::json!({
            "file_path": body.file_path,
            "dna": dna,
            "dimensions": dna.len(),
            "has_dna": true,
        }))),
        None => Ok(Json(serde_json::json!({
            "file_path": body.file_path,
            "dna": null,
            "dimensions": 0,
            "has_dna": false,
            "hint": "Run project sync + analytics first to compute structural DNA",
        }))),
    }
}

/// POST /api/code/structural-twins
///
/// Finds files structurally similar to a target file within the same project.
/// Uses multi-signal fusion (fingerprint cosine + WL hash match + file name
/// similarity + size similarity) with backward-compatible DNA fallback.
/// Returns ranked results sorted by descending similarity.
pub async fn find_structural_twins(
    State(state): State<OrchestratorState>,
    Json(body): Json<StructuralDnaBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::{compute_multi_signal_similarity, FileSignals};

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    // Resolve relative path to absolute
    let resolved_path = if !body.file_path.starts_with('/') {
        let expanded = crate::expand_tilde(&project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &body.file_path)
    } else {
        body.file_path.clone()
    };

    // Try fingerprint-based multi-signal comparison first
    let file_signals = state
        .orchestrator
        .neo4j()
        .get_project_file_signals(&project.id.to_string())
        .await?;

    if !file_signals.is_empty() {
        // Find source file signals
        let source_record = match file_signals.iter().find(|r| r.path == resolved_path) {
            Some(r) => r,
            None => {
                return Ok(Json(serde_json::json!({
                    "file_path": body.file_path,
                    "twins": [],
                    "total": 0,
                    "hint": format!("File '{}' has no structural fingerprint", body.file_path),
                })));
            }
        };

        let source_file = FileSignals {
            path: source_record.path.clone(),
            fingerprint: source_record.fingerprint.clone(),
            wl_hash: if source_record.wl_hash != 0 {
                Some(source_record.wl_hash)
            } else {
                None
            },
            function_count: source_record.function_count,
        };

        let top_n = body.top_n.unwrap_or(10);

        // Compute multi-signal similarities (skip self)
        let mut twins: Vec<serde_json::Value> = file_signals
            .iter()
            .filter(|r| r.path != resolved_path)
            .map(|record| {
                let target_file = FileSignals {
                    path: record.path.clone(),
                    fingerprint: record.fingerprint.clone(),
                    wl_hash: if record.wl_hash != 0 {
                        Some(record.wl_hash)
                    } else {
                        None
                    },
                    function_count: record.function_count,
                };
                let sim = compute_multi_signal_similarity(&source_file, &target_file);
                serde_json::json!({
                    "file_path": record.path,
                    "similarity": sim.similarity,
                    "signals": {
                        "fingerprint_similarity": sim.signals.fingerprint_similarity,
                        "wl_hash_match": sim.signals.wl_hash_match,
                        "name_similarity": sim.signals.name_similarity,
                        "size_similarity": sim.signals.size_similarity,
                    },
                    "shared_role": sim.shared_role,
                })
            })
            .collect();

        // Sort by descending similarity
        twins.sort_by(|a, b| {
            let sa = a["similarity"].as_f64().unwrap_or(0.0);
            let sb = b["similarity"].as_f64().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let total = twins.len();
        twins.truncate(top_n);

        return Ok(Json(serde_json::json!({
            "file_path": body.file_path,
            "twins": twins,
            "total": total,
            "returned": twins.len(),
            "method": "fingerprint_v2",
        })));
    }

    // Fallback: DNA-based cosine similarity (backward compatibility)
    use crate::graph::algorithms::cosine_similarity;

    let all_dna = state
        .orchestrator
        .neo4j()
        .get_project_structural_dna(&project.id.to_string())
        .await?;

    if all_dna.is_empty() {
        return Ok(Json(serde_json::json!({
            "file_path": body.file_path,
            "twins": [],
            "total": 0,
            "hint": "No structural data found. Run project sync + analytics first.",
        })));
    }

    let dna_map: std::collections::HashMap<String, Vec<f64>> = all_dna.into_iter().collect();

    let target_dna = match dna_map.get(&resolved_path) {
        Some(dna) => dna,
        None => {
            return Ok(Json(serde_json::json!({
                "file_path": body.file_path,
                "twins": [],
                "total": 0,
                "hint": format!("File '{}' has no structural DNA", body.file_path),
            })));
        }
    };

    let top_n = body.top_n.unwrap_or(10);

    let mut twins: Vec<serde_json::Value> = dna_map
        .iter()
        .filter(|(path, _)| path.as_str() != resolved_path)
        .map(|(path, dna)| {
            let similarity = cosine_similarity(target_dna, dna);
            serde_json::json!({
                "file_path": path,
                "similarity": similarity,
            })
        })
        .collect();

    twins.sort_by(|a, b| {
        let sa = a["similarity"].as_f64().unwrap_or(0.0);
        let sb = b["similarity"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = twins.len();
    twins.truncate(top_n);

    Ok(Json(serde_json::json!({
        "file_path": body.file_path,
        "twins": twins,
        "total": total,
        "returned": twins.len(),
        "method": "dna_legacy",
    })))
}

/// Request body for DNA clustering endpoint.
#[derive(Deserialize)]
pub struct ClusterDnaBody {
    pub project_slug: String,
    /// Number of clusters (default 5)
    pub n_clusters: Option<usize>,
}

/// POST /api/code/structural-clusters
///
/// Performs K-means clustering on structural DNA vectors to discover
/// architectural roles (handlers, models, services, etc.) within a project.
pub async fn cluster_dna(
    State(state): State<OrchestratorState>,
    Json(body): Json<ClusterDnaBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::cluster_dna_vectors;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let all_dna = state
        .orchestrator
        .neo4j()
        .get_project_structural_dna(&project.id.to_string())
        .await?;

    if all_dna.is_empty() {
        return Ok(Json(serde_json::json!({
            "clusters": [],
            "total_files": 0,
            "hint": "No structural DNA found. Run project sync + analytics first.",
        })));
    }

    let dna_map: std::collections::HashMap<String, Vec<f64>> = all_dna.into_iter().collect();

    let n_clusters = body.n_clusters.unwrap_or(5).min(dna_map.len());
    let clusters = cluster_dna_vectors(&dna_map, n_clusters);

    let total_files: usize = clusters.iter().map(|c| c.members.len()).sum();

    Ok(Json(serde_json::json!({
        "clusters": clusters,
        "total_files": total_files,
        "n_clusters": clusters.len(),
    })))
}

/// Request body for cross-project structural twins.
#[derive(Deserialize)]
pub struct CrossProjectTwinsBody {
    pub workspace_slug: String,
    pub source_project_slug: String,
    pub file_path: String,
    /// Max results (default 10)
    pub top_n: Option<usize>,
}

/// POST /api/code/structural-twins/cross-project
///
/// Finds structurally similar files across other projects in the same workspace.
/// Uses multi-signal fusion (fingerprint cosine + WL hash match + file name
/// similarity + size similarity) instead of raw DNA cosine.
///
/// Enables knowledge transfer: notes from a twin file in project B can be
/// suggested for the source file in project A.
pub async fn find_cross_project_twins(
    State(state): State<OrchestratorState>,
    Json(body): Json<CrossProjectTwinsBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::{compute_multi_signal_similarity, FileSignals};

    let neo4j = state.orchestrator.neo4j();

    // Resolve workspace
    let workspace = neo4j
        .get_workspace_by_slug(&body.workspace_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Workspace '{}' not found", body.workspace_slug))
        })?;

    // Get all projects in workspace
    let projects = neo4j.list_workspace_projects(workspace.id).await?;

    // Resolve source project
    let source_project = neo4j
        .get_project_by_slug(&body.source_project_slug)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!("Project '{}' not found", body.source_project_slug))
        })?;

    // Resolve relative path to absolute
    let resolved_path = if !body.file_path.starts_with('/') {
        let expanded = crate::expand_tilde(&source_project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &body.file_path)
    } else {
        body.file_path.clone()
    };

    // Get source project file signals (fingerprints + WL hashes + function counts)
    let source_signals = neo4j
        .get_project_file_signals(&source_project.id.to_string())
        .await?;

    // Find the source file's signals
    let source_record = match source_signals.iter().find(|r| r.path == resolved_path) {
        Some(r) => r,
        None => {
            return Ok(Json(serde_json::json!({
                "file_path": body.file_path,
                "twins": [],
                "total": 0,
                "hint": format!("File '{}' has no structural fingerprint in project '{}'. Run analytics first.", body.file_path, body.source_project_slug),
            })));
        }
    };

    let source_file = FileSignals {
        path: source_record.path.clone(),
        fingerprint: source_record.fingerprint.clone(),
        wl_hash: if source_record.wl_hash != 0 {
            Some(source_record.wl_hash)
        } else {
            None
        },
        function_count: source_record.function_count,
    };

    // Detect source language (extension) for hard filter
    let source_ext = resolved_path.rsplit('.').next().unwrap_or("");

    let top_n = body.top_n.unwrap_or(10);
    let mut all_twins: Vec<serde_json::Value> = Vec::new();

    // Search across all other projects using multi-signal fusion
    for project in &projects {
        if project.id == source_project.id {
            continue; // Skip source project
        }

        let other_signals = neo4j
            .get_project_file_signals(&project.id.to_string())
            .await?;

        for record in &other_signals {
            // Language hard filter: skip files with different extensions
            let target_ext = record.path.rsplit('.').next().unwrap_or("");
            if target_ext != source_ext {
                continue;
            }

            let target_file = FileSignals {
                path: record.path.clone(),
                fingerprint: record.fingerprint.clone(),
                wl_hash: if record.wl_hash != 0 {
                    Some(record.wl_hash)
                } else {
                    None
                },
                function_count: record.function_count,
            };

            let sim = compute_multi_signal_similarity(&source_file, &target_file);

            all_twins.push(serde_json::json!({
                "file_path": record.path,
                "project_slug": project.slug,
                "project_name": project.name,
                "similarity": sim.similarity,
                "signals": {
                    "fingerprint_similarity": sim.signals.fingerprint_similarity,
                    "wl_hash_match": sim.signals.wl_hash_match,
                    "name_similarity": sim.signals.name_similarity,
                    "size_similarity": sim.signals.size_similarity,
                },
                "shared_role": sim.shared_role,
            }));
        }
    }

    // Sort by descending fused similarity
    all_twins.sort_by(|a, b| {
        let sa = a["similarity"].as_f64().unwrap_or(0.0);
        let sb = b["similarity"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = all_twins.len();
    all_twins.truncate(top_n);

    Ok(Json(serde_json::json!({
        "file_path": body.file_path,
        "source_project": body.source_project_slug,
        "twins": all_twins,
        "total": total,
        "returned": all_twins.len(),
    })))
}

// ============================================================================
// Link Prediction
// ============================================================================

/// Request body for predict_missing_links endpoint.
#[derive(Debug, Deserialize)]
pub struct PredictMissingLinksBody {
    pub project_slug: String,
    pub top_n: Option<usize>,
    pub min_plausibility: Option<f64>,
}

/// POST /api/code/predict-links
///
/// Suggests the top-N most plausible missing links in the project's file graph.
/// Uses 5 signals (Jaccard, co-change, proximity, Adamic-Adar, DNA similarity)
/// to score candidate pairs at distance 2-3 that are not directly connected.
pub async fn predict_missing_links(
    State(state): State<OrchestratorState>,
    Json(body): Json<PredictMissingLinksBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::{extract_co_change_data, suggest_missing_links};
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let project_id = project.id.to_string();

    // Get structural DNA (optional, used as signal if available)
    let all_dna = state
        .orchestrator
        .neo4j()
        .get_project_structural_dna(&project_id)
        .await?;

    let dna_map: std::collections::HashMap<String, Vec<f64>> = all_dna.into_iter().collect();
    let dna_ref = if dna_map.is_empty() {
        None
    } else {
        Some(&dna_map)
    };

    // Extract file graph
    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    // Extract co-change data from CoChanged edges
    let co_change = extract_co_change_data(&graph);

    let top_n = body.top_n.unwrap_or(20);
    let min_plausibility = body.min_plausibility.unwrap_or(0.0);

    let predictions = suggest_missing_links(&graph, &co_change, dna_ref, top_n, min_plausibility);

    Ok(Json(serde_json::json!({
        "predictions": predictions,
        "total": predictions.len(),
        "top_n": top_n,
        "min_plausibility": min_plausibility,
    })))
}

/// Request body for check_link_plausibility endpoint.
#[derive(Debug, Deserialize)]
pub struct CheckLinkPlausibilityBody {
    pub project_slug: String,
    pub source: String,
    pub target: String,
}

/// POST /api/code/link-plausibility
///
/// Checks how plausible a specific link between two nodes would be.
/// Returns a single LinkPrediction with the combined score and individual
/// signal values (Jaccard, co-change, proximity, Adamic-Adar, DNA similarity).
pub async fn check_link_plausibility(
    State(state): State<OrchestratorState>,
    Json(body): Json<CheckLinkPlausibilityBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::{extract_co_change_data, link_plausibility};
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let project_id = project.id.to_string();

    // Get structural DNA (optional)
    let all_dna = state
        .orchestrator
        .neo4j()
        .get_project_structural_dna(&project_id)
        .await?;

    let dna_map: std::collections::HashMap<String, Vec<f64>> = all_dna.into_iter().collect();
    let dna_ref = if dna_map.is_empty() {
        None
    } else {
        Some(&dna_map)
    };

    // Extract file graph
    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    // Look up source and target in the graph
    let source_idx = graph.id_to_index.get(&body.source).ok_or_else(|| {
        AppError::NotFound(format!("Source node '{}' not found in graph", body.source))
    })?;
    let target_idx = graph.id_to_index.get(&body.target).ok_or_else(|| {
        AppError::NotFound(format!("Target node '{}' not found in graph", body.target))
    })?;

    // Extract co-change data
    let co_change = extract_co_change_data(&graph);

    let prediction = link_plausibility(&graph, *source_idx, *target_idx, &co_change, dna_ref);

    Ok(Json(serde_json::json!(prediction)))
}

// ============================================================================
// Stress Testing (Plan 5)
// ============================================================================

/// Request body for stress_test_node endpoint.
#[derive(Debug, Deserialize)]
pub struct StressTestNodeBody {
    pub project_slug: String,
    pub target_id: String,
}

/// POST /api/code/stress-test-node
///
/// Simulates removing a node from the project's file graph and measures the
/// impact: orphaned nodes, blast radius, resilience score, etc.
pub async fn stress_test_node(
    State(state): State<OrchestratorState>,
    Json(body): Json<StressTestNodeBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::stress_test_node_removal;
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    let result = stress_test_node_removal(&graph, &body.target_id).ok_or_else(|| {
        AppError::NotFound(format!(
            "Target node '{}' not found in graph",
            body.target_id
        ))
    })?;

    Ok(Json(serde_json::json!(result)))
}

/// Request body for stress_test_edge endpoint.
#[derive(Debug, Deserialize)]
pub struct StressTestEdgeBody {
    pub project_slug: String,
    pub from_id: String,
    pub to_id: String,
}

/// POST /api/code/stress-test-edge
///
/// Simulates removing an edge from the project's file graph and measures
/// whether it is a bridge (increases connected components).
pub async fn stress_test_edge(
    State(state): State<OrchestratorState>,
    Json(body): Json<StressTestEdgeBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::stress_test_edge_removal;
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    let result = stress_test_edge_removal(&graph, &body.from_id, &body.to_id).ok_or_else(|| {
        AppError::NotFound(format!(
            "Edge '{}' -> '{}' not found in graph",
            body.from_id, body.to_id
        ))
    })?;

    Ok(Json(serde_json::json!(result)))
}

/// Request body for stress_test_cascade endpoint.
#[derive(Debug, Deserialize)]
pub struct StressTestCascadeBody {
    pub project_slug: String,
    pub target_id: String,
    pub max_iterations: Option<usize>,
}

/// POST /api/code/stress-test-cascade
///
/// Simulates a cascading removal starting from a target node: removes the node,
/// then iteratively removes nodes whose incoming dependencies are all removed,
/// until no new orphans appear or max_iterations is reached.
pub async fn stress_test_cascade(
    State(state): State<OrchestratorState>,
    Json(body): Json<StressTestCascadeBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::stress_test_cascade;
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    let max_iterations = body.max_iterations.unwrap_or(10);

    let result = stress_test_cascade(&graph, &body.target_id, max_iterations).ok_or_else(|| {
        AppError::NotFound(format!(
            "Target node '{}' not found in graph",
            body.target_id
        ))
    })?;

    Ok(Json(serde_json::json!(result)))
}

/// Request body for find_bridges endpoint.
#[derive(Debug, Deserialize)]
pub struct FindBridgesBody {
    pub project_slug: String,
}

/// POST /api/code/find-bridges
///
/// Finds all bridge edges in the project's file graph. A bridge is an edge
/// whose removal increases the number of connected components.
pub async fn find_bridges(
    State(state): State<OrchestratorState>,
    Json(body): Json<FindBridgesBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::algorithms::find_bridges;
    use crate::graph::extraction::GraphExtractor;

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&body.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", body.project_slug)))?;

    let extractor = GraphExtractor::new(state.orchestrator.neo4j_arc());
    let graph = extractor.extract_file_graph(project.id).await?;

    let bridges = find_bridges(&graph);

    Ok(Json(serde_json::json!({
        "bridges": bridges,
        "total": bridges.len(),
    })))
}

// ============================================================================
// Context Cards (GraIL Plan 8)
// ============================================================================

#[derive(Deserialize)]
pub struct ContextCardQuery {
    /// File path (absolute or relative — resolved with project root)
    pub path: String,
    /// Project slug (required for scoping)
    pub project_slug: String,
}

/// Get pre-computed context card for a single file.
///
/// Returns the cached cc_* properties from Neo4j. If `cc_version == -1`,
/// the card is stale and should be refreshed via analytics re-run.
pub async fn get_context_card(
    State(state): State<OrchestratorState>,
    Query(query): Query<ContextCardQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;

    // Resolve relative path to absolute
    let path = if !query.path.starts_with('/') {
        let expanded = crate::expand_tilde(&project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &query.path)
    } else {
        query.path.clone()
    };

    let card = state
        .orchestrator
        .neo4j()
        .get_context_card(&path, &project.id.to_string())
        .await?;

    match card {
        Some(c) => Ok(Json(serde_json::to_value(c).unwrap_or_default())),
        None => Ok(Json(serde_json::json!({
            "path": path,
            "error": "no_context_card",
            "message": "No context card found. Run analytics (sync project) to compute context cards."
        }))),
    }
}

#[derive(Deserialize)]
pub struct RefreshContextCardsBody {
    /// Project slug (required)
    pub project_slug: String,
}

/// Force refresh of all context cards for a project by triggering analytics.
///
/// This triggers the analytics debouncer which will re-run compute_all
/// and persist updated context cards.
pub async fn refresh_context_cards(
    State(state): State<OrchestratorState>,
    Json(query): Json<RefreshContextCardsBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;

    // Trigger analytics debouncer (will recompute all cards)
    state.orchestrator.analytics_debouncer().trigger(project.id);

    Ok(Json(serde_json::json!({
        "status": "triggered",
        "project_slug": query.project_slug,
        "message": "Analytics refresh triggered. Context cards will be recomputed shortly."
    })))
}

// ============================================================================
// WL Fingerprint & Isomorphic Groups (Plan 7)
// ============================================================================

#[derive(Deserialize)]
pub struct FingerprintQuery {
    /// File path (required)
    pub path: String,
    /// Project slug (required)
    pub project_slug: String,
}

/// Get the structural fingerprint for a single file.
///
/// Returns the 17-dim universal fingerprint vector with dimension labels,
/// plus the WL hash and legacy structural DNA for backward compatibility.
pub async fn get_fingerprint(
    State(state): State<OrchestratorState>,
    Query(query): Query<FingerprintQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::graph::models::{FINGERPRINT_DIMS, FINGERPRINT_LABELS};

    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;

    // Resolve relative path
    let path = if !query.path.starts_with('/') {
        let expanded = crate::expand_tilde(&project.root_path);
        format!("{}/{}", expanded.trim_end_matches('/'), &query.path)
    } else {
        query.path.clone()
    };

    let card = state
        .orchestrator
        .neo4j()
        .get_context_card(&path, &project.id.to_string())
        .await?;

    match card {
        Some(c) => {
            // Build labeled fingerprint dimensions for readability
            let labeled_fingerprint: serde_json::Value =
                if c.cc_fingerprint.len() == FINGERPRINT_DIMS {
                    let dims: Vec<serde_json::Value> = FINGERPRINT_LABELS
                        .iter()
                        .zip(c.cc_fingerprint.iter())
                        .map(|(label, &val)| serde_json::json!({ "label": label, "value": val }))
                        .collect();
                    serde_json::json!(dims)
                } else {
                    serde_json::json!(null)
                };

            Ok(Json(serde_json::json!({
                "path": c.path,
                "project_slug": query.project_slug,
                "fingerprint": c.cc_fingerprint,
                "fingerprint_dims": FINGERPRINT_DIMS,
                "fingerprint_labeled": labeled_fingerprint,
                "wl_hash": c.cc_wl_hash,
                "structural_dna": c.cc_structural_dna,
            })))
        }
        None => Ok(Json(serde_json::json!({
            "path": path,
            "project_slug": query.project_slug,
            "fingerprint": null,
            "wl_hash": null,
            "message": "No fingerprint computed yet. Run analytics or refresh context cards.",
        }))),
    }
}

#[derive(Deserialize)]
pub struct IsomorphicQuery {
    /// Project slug (required)
    pub project_slug: String,
    /// Minimum group size (default: 2)
    pub min_group_size: Option<usize>,
}

/// Find groups of files with identical WL subgraph hash (isomorphic neighborhoods).
pub async fn find_isomorphic(
    State(state): State<OrchestratorState>,
    Query(query): Query<IsomorphicQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;

    let min_size = query.min_group_size.unwrap_or(2);
    let groups = state
        .orchestrator
        .neo4j()
        .find_isomorphic_groups(&project.id.to_string(), min_size)
        .await?;

    Ok(Json(serde_json::json!({
        "project_slug": query.project_slug,
        "min_group_size": min_size,
        "groups_count": groups.len(),
        "groups": groups,
    })))
}

// ============================================================================
// Structural Templates
// ============================================================================

#[derive(Deserialize)]
pub struct StructuralTemplateQuery {
    pub project_slug: String,
    /// Minimum number of files sharing the same WL hash to form a template (default: 3)
    pub min_occurrences: Option<usize>,
}

/// GET /api/code/structural-templates — Suggest reusable structural templates
/// from isomorphic groups (files sharing the same WL hash fingerprint).
pub async fn suggest_structural_templates(
    State(state): State<OrchestratorState>,
    Query(query): Query<StructuralTemplateQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&query.project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", query.project_slug)))?;

    let min_occ = query.min_occurrences.unwrap_or(3);

    // Get isomorphic groups with min_size = min_occurrences
    let groups = state
        .orchestrator
        .neo4j()
        .find_isomorphic_groups(&project.id.to_string(), min_occ)
        .await?;

    // For each group, try to derive a description from common path patterns and structural DNA
    let mut templates: Vec<crate::graph::models::StructuralTemplate> = Vec::new();

    for group in groups {
        // Derive description from common path pattern
        let description = derive_pattern_description(&group.members);

        // Find common DNA prefix from context cards
        let exemplars: Vec<String> = group.members.iter().take(5).cloned().collect();

        // Try to get context cards for DNA info — convert Vec<f64> to string for comparison
        let mut dna_strings: Vec<String> = Vec::new();
        for path in exemplars.iter().take(3) {
            if let Ok(Some(card)) = state
                .orchestrator
                .neo4j()
                .get_context_card(path, &project.id.to_string())
                .await
            {
                if !card.cc_structural_dna.is_empty() {
                    // Format DNA as compact string: "0.12,0.45,0.78,..."
                    let dna_str: String = card
                        .cc_structural_dna
                        .iter()
                        .map(|v| format!("{:.2}", v))
                        .collect::<Vec<_>>()
                        .join(",");
                    dna_strings.push(dna_str);
                }
            }
        }

        let common_dna_prefix = if dna_strings.len() >= 2 {
            find_common_prefix(&dna_strings)
        } else {
            None
        };

        templates.push(crate::graph::models::StructuralTemplate {
            wl_hash: group.wl_hash,
            occurrences: group.size,
            exemplars,
            description,
            common_dna_prefix,
        });
    }

    // Sort by occurrences descending
    templates.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));

    Ok(Json(serde_json::json!({
        "project_slug": query.project_slug,
        "min_occurrences": min_occ,
        "template_count": templates.len(),
        "templates": templates,
    })))
}

/// Derive a human-readable description from file path patterns
fn derive_pattern_description(paths: &[String]) -> String {
    if paths.is_empty() {
        return "Unknown pattern".to_string();
    }

    // Find common directory
    let parts: Vec<Vec<&str>> = paths.iter().map(|p| p.split('/').collect()).collect();

    // Find common prefix length
    let min_len = parts.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut common_depth = 0;
    for i in 0..min_len {
        if parts.iter().all(|p| p[i] == parts[0][i]) {
            common_depth = i + 1;
        } else {
            break;
        }
    }

    // Find common file suffixes (e.g., all end in "_handler.rs", "_test.rs")
    let filenames: Vec<&str> = paths.iter().filter_map(|p| p.rsplit('/').next()).collect();
    let common_suffix = find_common_filename_suffix(&filenames);

    let dir_prefix = if common_depth > 0 {
        parts[0][..common_depth].join("/")
    } else {
        String::new()
    };

    match (dir_prefix.is_empty(), common_suffix.is_empty()) {
        (false, false) => format!(
            "Files in {dir_prefix}/ matching *{common_suffix} ({} files)",
            paths.len()
        ),
        (false, true) => format!(
            "Files in {dir_prefix}/ with identical topology ({} files)",
            paths.len()
        ),
        (true, false) => format!(
            "Files matching *{common_suffix} with identical topology ({} files)",
            paths.len()
        ),
        (true, true) => format!("Structurally identical files ({} files)", paths.len()),
    }
}

/// Find common suffix in filenames (e.g., "_handler.rs", "_test.rs")
fn find_common_filename_suffix(names: &[&str]) -> String {
    if names.len() < 2 {
        return String::new();
    }
    let first: Vec<char> = names[0].chars().rev().collect();
    let mut common_len = 0;
    for i in 0..first.len() {
        if names.iter().all(|n| {
            let chars: Vec<char> = n.chars().rev().collect();
            chars.len() > i && chars[i] == first[i]
        }) {
            common_len = i + 1;
        } else {
            break;
        }
    }
    if common_len > 3 {
        // Only return if meaningful
        first[..common_len].iter().rev().collect()
    } else {
        String::new()
    }
}

/// Find common prefix among DNA strings
fn find_common_prefix(values: &[String]) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    let first = &values[0];
    let mut prefix_len = first.len();
    for v in &values[1..] {
        prefix_len = prefix_len.min(v.len());
        for (i, (a, b)) in first.chars().zip(v.chars()).enumerate() {
            if a != b {
                prefix_len = prefix_len.min(i);
                break;
            }
        }
    }
    if prefix_len > 5 {
        Some(first[..prefix_len].to_string())
    } else {
        None
    }
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
            registry_remote_url: None,
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
            registry_remote_url: None,
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
        // Response is now a CodeSearchResult wrapper (Plan 10)
        assert!(json.is_object());
        assert!(json["hits"].is_array());
        assert_eq!(json["hits"].as_array().unwrap().len(), 0);
        assert!(json["ranked"]["items"].is_array());
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
        // Response is now a CodeSearchResult wrapper (Plan 10)
        let results = json["hits"].as_array().unwrap();
        assert!(!results.is_empty());
        // Verify SearchHit<CodeDocument> shape: { document, score }
        let first = &results[0];
        assert!(first["document"].is_object());
        assert!(first["score"].is_number());
        assert_eq!(first["document"]["path"], "src/main.rs");
        // Verify ranked view is present
        assert!(json["ranked"]["items"].is_array());
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
        let results = json["hits"].as_array().unwrap();
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
        assert_eq!(json["hits"].as_array().unwrap().len(), 0);
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
            last_co_change_computed_at: None,
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
            registry_remote_url: None,
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
        let results = json["hits"].as_array().unwrap();
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
            last_co_change_computed_at: None,
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
            registry_remote_url: None,
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
            last_co_change_computed_at: None,
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
            registry_remote_url: None,
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
            last_co_change_computed_at: None,
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
            registry_remote_url: None,
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
            last_co_change_computed_at: None,
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
            registry_remote_url: None,
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

    // ====================================================================
    // GET /api/code/symbols/{file_path} — get_file_symbols
    // ====================================================================

    #[tokio::test]
    async fn test_get_file_symbols_not_found() {
        // File not in graph → 404
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/symbols/nonexistent.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_file_symbols_empty_file() {
        // File exists in graph but has no functions or structs
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(auth_get("/api/code/symbols/src/main.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["path"], "src/main.rs");
        assert_eq!(json["language"], "rust");
        assert!(json["functions"].as_array().unwrap().is_empty());
        assert!(json["structs"].as_array().unwrap().is_empty());
        assert!(json["imports"].as_array().unwrap().is_empty());
    }

    /// Build a test router with a file that has functions, structs, and imports
    async fn test_app_with_symbols() -> axum::Router {
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FunctionNode, ImportNode, Parameter, StructNode, Visibility};
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();

        // Seed file
        let file = FileNode {
            path: "src/handler.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: None,
        };
        graph.upsert_file(&file).await.unwrap();

        // Seed a function
        let func = FunctionNode {
            name: "handle_request".to_string(),
            visibility: Visibility::Public,
            params: vec![Parameter {
                name: "req".to_string(),
                type_name: Some("Request".to_string()),
            }],
            return_type: Some("Response".to_string()),
            generics: vec![],
            is_async: true,
            is_unsafe: false,
            complexity: 3,
            file_path: "src/handler.rs".to_string(),
            line_start: 10,
            line_end: 30,
            docstring: Some("Handle incoming request".to_string()),
        };
        graph.upsert_function(&func).await.unwrap();

        // Seed a struct
        let s = StructNode {
            name: "Config".to_string(),
            visibility: Visibility::Public,
            generics: vec![],
            file_path: "src/handler.rs".to_string(),
            line_start: 5,
            line_end: 8,
            docstring: Some("App config".to_string()),
            parent_class: None,
            interfaces: vec![],
        };
        graph.upsert_struct(&s).await.unwrap();

        // Seed an import
        let imp = ImportNode {
            path: "std::io".to_string(),
            alias: None,
            items: vec![],
            file_path: "src/handler.rs".to_string(),
            line: 1,
        };
        graph.upsert_import(&imp).await.unwrap();

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
            registry_remote_url: None,
        });
        create_router(state)
    }

    #[tokio::test]
    async fn test_get_file_symbols_with_data() {
        let app = test_app_with_symbols().await;
        let resp = app
            .oneshot(auth_get("/api/code/symbols/src/handler.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["path"], "src/handler.rs");
        assert_eq!(json["language"], "rust");

        // Check function
        let functions = json["functions"].as_array().unwrap();
        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0]["name"], "handle_request");
        assert!(functions[0]["is_async"].as_bool().unwrap());
        assert!(functions[0]["is_public"].as_bool().unwrap());
        assert_eq!(functions[0]["line"], 10);
        assert_eq!(functions[0]["complexity"], 3);
        assert_eq!(functions[0]["docstring"], "Handle incoming request");
        let sig = functions[0]["signature"].as_str().unwrap();
        assert!(sig.contains("handle_request"), "signature: {}", sig);
        assert!(sig.contains("req: Request"), "signature: {}", sig);
        assert!(sig.contains("-> Response"), "signature: {}", sig);

        // Check struct
        let structs = json["structs"].as_array().unwrap();
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0]["name"], "Config");
        assert!(structs[0]["is_public"].as_bool().unwrap());
        assert_eq!(structs[0]["line"], 5);

        // Check imports
        let imports = json["imports"].as_array().unwrap();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0], "std::io");
    }

    // ====================================================================
    // GET /api/code/references — find_references
    // ====================================================================

    #[tokio::test]
    async fn test_find_references_empty() {
        // No references for a symbol that doesn't exist
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/references?symbol=nonexistent_function"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Response is now a FindReferencesResult wrapper (Plan 10)
        assert!(json.is_object());
        assert!(json["references"].is_array());
        assert_eq!(json["references"].as_array().unwrap().len(), 0);
        assert!(json["ranked"]["items"].is_array());
        assert_eq!(json["ranked"]["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_find_references_with_callers() {
        // Seed call relationships: caller_fn calls "target_fn"
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FunctionNode, Visibility};
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();

        // Seed a caller function
        let caller = FunctionNode {
            name: "caller_fn".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/caller.rs".to_string(),
            line_start: 5,
            line_end: 15,
            docstring: None,
        };
        graph.upsert_function(&caller).await.unwrap();

        // Seed the call relationship: caller_fn calls target_fn
        graph.call_relationships.write().await.insert(
            "src/caller.rs::caller_fn".to_string(),
            vec!["target_fn".to_string()],
        );

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
            registry_remote_url: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get("/api/code/references?symbol=target_fn"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Response is now a FindReferencesResult wrapper (Plan 10)
        let refs = json["references"].as_array().unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0]["file_path"], "src/caller.rs");
        assert_eq!(refs[0]["reference_type"], "call");
        assert_eq!(refs[0]["line"], 5);
        // Verify ranked view is also present
        let ranked_items = json["ranked"]["items"].as_array().unwrap();
        assert_eq!(ranked_items.len(), 1);
        assert_eq!(ranked_items[0]["file_path"], "src/caller.rs");
        assert_eq!(ranked_items[0]["rank"], 1);
        assert_eq!(ranked_items[0]["score"], 1.0); // call type → 1.0
    }

    #[tokio::test]
    async fn test_find_references_missing_symbol_param() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/references?limit=5"))
            .await
            .unwrap();

        // Missing required 'symbol' param → 400
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/code/dependencies/{file_path} — get_file_dependencies
    // ====================================================================

    #[tokio::test]
    async fn test_get_file_dependencies_empty() {
        // File with no import relationships → empty lists
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/dependencies/src/orphan.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["imports"].as_array().unwrap().is_empty());
        assert!(json["imported_by"].as_array().unwrap().is_empty());
        assert!(json["impact_radius"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_file_dependencies_with_imports() {
        // Use the existing test_app_with_imports() which has import relationships seeded
        let app = test_app_with_imports().await;
        let resp = app
            .oneshot(auth_get("/api/code/dependencies/src/handler.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // handler.rs imports lib.rs
        let imports = json["imports"].as_array().unwrap();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0]["path"], "src/lib.rs");
    }

    #[tokio::test]
    async fn test_get_file_dependencies_imported_by() {
        // lib.rs is imported by handler.rs, utils.rs, test_lib.rs
        let app = test_app_with_imports().await;
        let resp = app
            .oneshot(auth_get("/api/code/dependencies/src/lib.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // lib.rs is depended on by 3 files
        let imported_by = json["imported_by"].as_array().unwrap();
        assert_eq!(imported_by.len(), 3);
        let impact_radius = json["impact_radius"].as_array().unwrap();
        assert_eq!(impact_radius.len(), 3);
    }

    // ====================================================================
    // GET /api/code/callgraph — get_call_graph
    // ====================================================================

    #[tokio::test]
    async fn test_get_call_graph_empty() {
        // Function not in graph → returns empty callers/callees
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/callgraph?function=nonexistent_fn"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["name"], "nonexistent_fn");
        assert!(json["callers"].as_array().unwrap().is_empty());
        assert!(json["callees"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_call_graph_with_relationships() {
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FunctionNode, Visibility};
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();

        // Seed functions
        let caller = FunctionNode {
            name: "caller".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/a.rs".to_string(),
            line_start: 1,
            line_end: 10,
            docstring: None,
        };
        graph.upsert_function(&caller).await.unwrap();

        let target = FunctionNode {
            name: "target".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/b.rs".to_string(),
            line_start: 1,
            line_end: 10,
            docstring: None,
        };
        graph.upsert_function(&target).await.unwrap();

        // caller calls target; target calls "downstream"
        {
            let mut cr = graph.call_relationships.write().await;
            cr.insert("src/a.rs::caller".to_string(), vec!["target".to_string()]);
            cr.insert(
                "src/b.rs::target".to_string(),
                vec!["downstream".to_string()],
            );
        }

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
            registry_remote_url: None,
        });
        let app = create_router(state);

        // Test direction=both (default)
        let resp = app
            .oneshot(auth_get("/api/code/callgraph?function=target"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["name"], "target");
        // "caller" calls "target"
        let callers = json["callers"].as_array().unwrap();
        assert_eq!(callers.len(), 1, "callers: {:?}", callers);
        // "target" calls "downstream"
        let callees = json["callees"].as_array().unwrap();
        assert_eq!(callees.len(), 1, "callees: {:?}", callees);
        assert_eq!(callees[0], "downstream");
    }

    #[tokio::test]
    async fn test_get_call_graph_callers_only() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/callgraph?function=some_fn&direction=callers",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // With direction=callers, callees should be empty
        assert!(json["callees"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_call_graph_missing_function_param() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/callgraph?depth=2"))
            .await
            .unwrap();

        // Missing required 'function' param → 400
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/code/impact — analyze_impact additional coverage
    // ====================================================================

    #[tokio::test]
    async fn test_analyze_impact_file_not_in_graph() {
        // analyze_impact for a file with no relationships → empty lists, low risk
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/impact?target=src/unknown.rs"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["target"], "src/unknown.rs");
        assert!(json["directly_affected"].as_array().unwrap().is_empty());
        assert!(json["transitively_affected"].as_array().unwrap().is_empty());
        assert!(json["test_files_affected"].as_array().unwrap().is_empty());
        assert_eq!(json["caller_count"], 0);
        assert_eq!(json["risk_level"], "low");
    }

    #[tokio::test]
    async fn test_analyze_impact_function_mode() {
        // analyze_impact with target_type=function should use callers path
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::models::{FunctionNode, Visibility};
        use crate::neo4j::traits::GraphStore;
        use crate::test_helpers::mock_app_state_with;

        let graph = MockGraphStore::new();

        // Seed a target function
        let target_fn = FunctionNode {
            name: "do_work".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 2,
            file_path: "src/worker.rs".to_string(),
            line_start: 10,
            line_end: 20,
            docstring: None,
        };
        graph.upsert_function(&target_fn).await.unwrap();

        // Seed a caller function
        let caller_fn = FunctionNode {
            name: "main".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: None,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 5,
            docstring: None,
        };
        graph.upsert_function(&caller_fn).await.unwrap();

        // main calls do_work
        graph
            .call_relationships
            .write()
            .await
            .insert("src/main.rs::main".to_string(), vec!["do_work".to_string()]);

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
            registry_remote_url: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(auth_get(
                "/api/code/impact?target=do_work&target_type=function",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["target"], "do_work");
        // caller_count comes from get_function_caller_count
        let caller_count = json["caller_count"].as_i64().unwrap();
        assert_eq!(caller_count, 1);
        // directly_affected should contain files of callers
        let direct = json["directly_affected"].as_array().unwrap();
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0], "src/main.rs");
    }

    // ====================================================================
    // POST /api/code/similar — find_similar_code
    // ====================================================================

    #[tokio::test]
    async fn test_find_similar_code_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/code/similar")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"snippet": "fn main()"}).to_string(),
                    ))
                    .unwrap(),
            )
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
    async fn test_find_similar_code_with_results() {
        let app = test_app_with_code().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/code/similar")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"snippet": "main entry point", "limit": 5}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let results = json.as_array().unwrap();
        assert!(!results.is_empty());
        // Verify SimilarCode shape
        assert!(results[0]["path"].is_string());
        assert!(results[0]["similarity"].is_number());
        assert!(results[0]["snippet"].is_string());
    }

    // ================================================================
    // Knowledge Fabric — Structural Analytics & Risk Tests
    // ================================================================

    /// Parse response body as JSON
    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ----------------------------------------------------------------
    // GET /api/code/communities
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_communities_valid_project() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/communities?project_slug=analytics-proj",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // May be empty or have communities depending on mock
        assert!(json["communities"].is_array());
    }

    #[tokio::test]
    async fn test_get_communities_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/communities?project_slug=nonexistent"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_communities_with_min_size() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/communities?project_slug=analytics-proj&min_size=5",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // GET /api/code/health
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_code_health_valid_project() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get("/api/code/health?project_slug=analytics-proj"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["god_functions"].is_array());
        assert!(json["orphan_files"].is_array());
    }

    #[tokio::test]
    async fn test_get_code_health_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/health?project_slug=nonexistent"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_code_health_custom_threshold() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/health?project_slug=analytics-proj&god_function_threshold=5",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // GET /api/code/node-importance
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_node_importance_not_in_mock() {
        // Mock get_node_gds_metrics returns None for all nodes,
        // so this should return 404
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/node-importance?project_slug=analytics-proj&node_path=src/main.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_node_importance_nonexistent_node() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/node-importance?project_slug=analytics-proj&node_path=nonexistent.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_node_importance_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/node-importance?project_slug=nonexistent&node_path=src/main.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ----------------------------------------------------------------
    // GET /api/code/hotspots — Change hotspots (churn)
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_change_hotspots_valid() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get("/api/code/hotspots?project_slug=analytics-proj"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["hotspots"].is_array());
        assert!(json["total_files"].is_number());
    }

    #[tokio::test]
    async fn test_get_change_hotspots_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/code/hotspots?project_slug=nonexistent"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_change_hotspots_with_limit() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/hotspots?project_slug=analytics-proj&limit=5",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // GET /api/code/knowledge-gaps — Knowledge density
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_knowledge_gaps_valid() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/knowledge-gaps?project_slug=analytics-proj",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["knowledge_gaps"].is_array());
        assert!(json["total_files"].is_number());
    }

    #[tokio::test]
    async fn test_get_knowledge_gaps_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/knowledge-gaps?project_slug=nonexistent",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_knowledge_gaps_with_limit() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/knowledge-gaps?project_slug=analytics-proj&limit=10",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // GET /api/code/risk-assessment — Composite risk score
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_risk_assessment_valid() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/risk-assessment?project_slug=analytics-proj",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["risk_files"].is_array());
        assert!(json["total_files"].is_number());
        assert!(json["summary"]["avg_risk_score"].is_number());
    }

    #[tokio::test]
    async fn test_get_risk_assessment_project_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/risk-assessment?project_slug=nonexistent",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_risk_assessment_with_limit() {
        let (app, _pid) = test_app_with_analytics().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/risk-assessment?project_slug=analytics-proj&limit=10",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ----------------------------------------------------------------
    // Deserialization tests for query structs
    // ----------------------------------------------------------------

    #[test]
    fn test_communities_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","min_size":3}"#;
        let q: CommunitiesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.min_size, Some(3));
    }

    #[test]
    fn test_communities_query_defaults() {
        let json = r#"{"project_slug":"my-proj"}"#;
        let q: CommunitiesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.min_size, None);
    }

    #[test]
    fn test_code_health_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","god_function_threshold":5}"#;
        let q: CodeHealthQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.god_function_threshold, Some(5));
    }

    #[test]
    fn test_hotspots_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","limit":10}"#;
        let q: HotspotsQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn test_knowledge_gaps_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","limit":15}"#;
        let q: KnowledgeGapsQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.limit, Some(15));
    }

    #[test]
    fn test_risk_assessment_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","limit":20}"#;
        let q: RiskAssessmentQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.limit, Some(20));
    }

    #[test]
    fn test_node_importance_query_deserialize() {
        let json = r#"{"project_slug":"my-proj","node_path":"src/main.rs","node_type":"file"}"#;
        let q: NodeImportanceQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.project_slug, "my-proj");
        assert_eq!(q.node_path, "src/main.rs");
        assert_eq!(q.node_type, Some("file".to_string()));
    }

    // ====================================================================
    // POST /api/code/processes/detect — nonexistent project
    // ====================================================================

    /// Create an authenticated POST request with JSON body
    fn auth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn test_detect_processes_404() {
        let app = test_app().await;
        let body = serde_json::json!({ "project_slug": "nonexistent-project" });
        let resp = app
            .oneshot(auth_post("/api/code/processes/detect", body))
            .await
            .unwrap();

        // Project not found should return 404
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // GET /api/code/class-hierarchy — max_depth clamping
    // ====================================================================

    #[tokio::test]
    async fn test_get_class_hierarchy_max_depth_clamp() {
        // Verify the handler doesn't panic when max_depth=100.
        // The handler clamps max_depth to 20 via .clamp(1, 20).
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/code/class-hierarchy?type_name=SomeClass&max_depth=100",
            ))
            .await
            .unwrap();

        // The handler should succeed (200) — the mock graph store returns an
        // empty hierarchy, so we just verify it doesn't panic or error out
        // due to the extreme max_depth value.
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Handler should succeed even with max_depth=100 (clamped to 20)"
        );
    }
}
