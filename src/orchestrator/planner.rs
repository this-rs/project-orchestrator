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
// Planner
// ============================================================================

/// Analyzes the knowledge graph to produce implementation plans
pub struct ImplementationPlanner {
    neo4j: Arc<dyn GraphStore>,
    meili: Arc<dyn SearchStore>,
    plan_manager: Arc<PlanManager>,
    note_manager: Arc<NoteManager>,
}

/// Maximum number of zones to consider
const MAX_ZONES: usize = 20;
/// Maximum depth for dependency expansion
const MAX_DEPENDENCY_DEPTH: u32 = 2;
/// Threshold for high risk (dependents count)
const HIGH_RISK_THRESHOLD: usize = 10;
/// Threshold for medium risk (dependents count)
const MEDIUM_RISK_THRESHOLD: usize = 3;

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
        }
    }

    // ========================================================================
    // Phase 1 — Zone identification (Task 2)
    // ========================================================================

    /// Resolve explicit entry points to RelevantZones.
    /// If the entry point looks like a file path (contains `/` or `.`), resolve via
    /// `get_file_symbol_names`. Otherwise treat it as a symbol name and use
    /// `find_symbol_references` to locate the file.
    async fn resolve_explicit_entries(
        &self,
        entry_points: &[String],
        project_id: Uuid,
    ) -> Result<Vec<RelevantZone>> {
        let mut zones = Vec::new();
        for entry in entry_points {
            if entry.contains('/') || entry.contains('.') {
                // File path — get symbols in this file
                match self.neo4j.get_file_symbol_names(entry).await {
                    Ok(symbols) => {
                        let mut functions = symbols.functions;
                        functions.extend(symbols.structs);
                        functions.extend(symbols.traits);
                        functions.extend(symbols.enums);
                        zones.push(RelevantZone {
                            file_path: entry.clone(),
                            functions,
                            relevance_score: 1.0,
                            source: ZoneSource::ExplicitEntry,
                        });
                    }
                    Err(_) => {
                        // File not in graph — still add it with no symbols
                        zones.push(RelevantZone {
                            file_path: entry.clone(),
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

    /// Fallback: search for relevant zones via semantic code search + note search.
    /// Uses `tokio::join!` to parallelize the two independent queries.
    async fn search_semantic_zones(
        &self,
        description: &str,
        project_slug: Option<&str>,
    ) -> Result<Vec<RelevantZone>> {
        let (code_results, note_results) = tokio::join!(
            self.meili
                .search_code_with_scores(description, 10, None, project_slug, None),
            self.meili
                .search_notes_with_filters(description, 10, project_slug, None, None, None),
        );

        let mut zones = Vec::new();

        // Code search results
        if let Ok(hits) = code_results {
            for hit in hits {
                zones.push(RelevantZone {
                    file_path: hit.document.path.clone(),
                    functions: hit.document.symbols.clone(),
                    relevance_score: hit.score.min(1.0),
                    source: ZoneSource::SemanticSearch,
                });
            }
        }

        // Note search results — extract file paths from anchor entities
        if let Ok(notes) = note_results {
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

        Ok(zones)
    }

    /// Orchestrate zone identification: explicit entries take priority,
    /// otherwise fall back to semantic search. Dedup by file_path (keep best score),
    /// sort by score descending, limit to MAX_ZONES.
    async fn identify_zones(&self, request: &PlanRequest) -> Result<Vec<RelevantZone>> {
        let zones = if let Some(ref entries) = request.entry_points {
            if !entries.is_empty() {
                self.resolve_explicit_entries(entries, request.project_id)
                    .await?
            } else {
                self.search_semantic_zones(&request.description, request.project_slug.as_deref())
                    .await?
            }
        } else {
            self.search_semantic_zones(&request.description, request.project_slug.as_deref())
                .await?
        };

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

    /// Expand dependencies for each zone: find dependent files (files that import
    /// this zone's file) and direct imports. Returns (dependents_map, imports_map).
    async fn expand_dependencies(
        &self,
        zones: &[RelevantZone],
        project_id: Uuid,
    ) -> Result<(HashMap<String, Vec<String>>, HashMap<String, Vec<String>>)> {
        let mut dependents_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut imports_map: HashMap<String, Vec<String>> = HashMap::new();

        for zone in zones {
            let (dependents, imports) = tokio::join!(
                self.neo4j.find_dependent_files(
                    &zone.file_path,
                    MAX_DEPENDENCY_DEPTH,
                    Some(project_id)
                ),
                self.neo4j.get_file_direct_imports(&zone.file_path),
            );

            if let Ok(deps) = dependents {
                dependents_map.insert(zone.file_path.clone(), deps);
            }
            if let Ok(imps) = imports {
                imports_map.insert(
                    zone.file_path.clone(),
                    imps.into_iter().map(|i| i.path).collect(),
                );
            }
        }

        Ok((dependents_map, imports_map))
    }

    /// Build the modification DAG from zones and their dependencies.
    /// Edge A→B means "A must be modified before B" (B depends on A).
    /// Cycles are detected and broken by skipping the back-edge.
    async fn build_dag(
        &self,
        zones: &[RelevantZone],
        dependents_map: &HashMap<String, Vec<String>>,
        project_id: Uuid,
    ) -> Result<ModificationDag> {
        let zone_files: HashSet<String> = zones.iter().map(|z| z.file_path.clone()).collect();

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

            // Fetch community_id from graph analytics (best-effort)
            let community_id = self
                .neo4j
                .get_node_analytics(&zone.file_path, "file")
                .await
                .ok()
                .flatten()
                .and_then(|a| a.community_id)
                .map(|c| c as u32);

            nodes.insert(
                zone.file_path.clone(),
                DagNode {
                    file_path: zone.file_path.clone(),
                    symbols,
                    risk,
                    test_files,
                    dependents_count,
                    community_id,
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

        // Step 2: Expand dependencies
        let (dependents_map, _imports_map) =
            self.expand_dependencies(&zones, request.project_id).await?;

        // Step 3: Build DAG
        let dag = self
            .build_dag(&zones, &dependents_map, request.project_id)
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
            .resolve_explicit_entries(&["src/models.rs".to_string()], pid)
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
            .resolve_explicit_entries(&["src/unknown.rs".to_string()], pid)
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

        let (dependents_map, _) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, pid)
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

        let (dependents_map, _) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, pid)
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

        let (dependents_map, _) = planner.expand_dependencies(&zones, pid).await.unwrap();
        let dag = planner
            .build_dag(&zones, &dependents_map, pid)
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
        };

        let plan = planner.plan_implementation(request).await.unwrap();
        // Should have a plan_id
        assert!(plan.plan_id.is_some());
        let plan_id_str = plan.plan_id.unwrap();
        assert!(!plan_id_str.is_empty());
        // Verify it's a valid UUID
        Uuid::parse_str(&plan_id_str).unwrap();
    }
}
