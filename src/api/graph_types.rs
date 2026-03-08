//! Shared graph types and helpers for project and workspace graph endpoints.
//!
//! Extracted from `project_handlers.rs` to enable reuse by `workspace_handlers.rs`.

use crate::api::handlers::AppError;
use crate::neo4j::models::ProjectNode;
use crate::neo4j::GraphStore;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

// ============================================================================
// Graph visualization types
// ============================================================================

/// Allowed layer names (whitelist per security constraint)
pub const VALID_LAYERS: &[&str] = &[
    "code",
    "knowledge",
    "fabric",
    "neural",
    "skills",
    "behavioral",
];

/// A node in the project graph visualization
#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    pub layer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
}

/// An edge in the project graph visualization
#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub layer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
}

/// A community cluster in the project graph
#[derive(Debug, Clone, Serialize)]
pub struct GraphCommunity {
    pub id: i64,
    pub label: String,
    pub file_count: usize,
    pub key_files: Vec<String>,
}

/// Per-layer statistics
#[derive(Debug, Clone, Serialize)]
pub struct LayerStats {
    pub nodes: usize,
    pub edges: usize,
}

/// Response for GET /api/projects/:slug/graph
#[derive(Debug, Clone, Serialize)]
pub struct ProjectGraphResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub communities: Vec<GraphCommunity>,
    pub stats: HashMap<String, LayerStats>,
}

/// Query parameters for the graph endpoint
#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    /// Comma-separated layers: code,knowledge,fabric,neural,skills,behavioral (default: code)
    pub layers: Option<String>,
    /// Filter nodes by community_id
    pub community: Option<i64>,
    /// Max nodes per layer (default: 5000)
    pub limit: Option<usize>,
}

/// Parse and validate the layers query parameter
pub fn parse_layers(layers_param: &Option<String>) -> Vec<String> {
    match layers_param {
        Some(s) => s
            .split(',')
            .map(|l| l.trim().to_lowercase())
            .filter(|l| VALID_LAYERS.contains(&l.as_str()))
            .collect(),
        None => vec!["code".to_string()],
    }
}

// ============================================================================
// Intelligence summary types
// ============================================================================

/// Code layer metrics
#[derive(Debug, Clone, Serialize)]
pub struct CodeLayerSummary {
    pub files: i64,
    pub functions: usize,
    pub communities: usize,
    pub hotspots: Vec<HotspotEntry>,
    pub orphans: usize,
}

/// A single hotspot entry
#[derive(Debug, Clone, Serialize)]
pub struct HotspotEntry {
    pub path: String,
    pub churn_score: f64,
}

/// Knowledge layer metrics
#[derive(Debug, Clone, Serialize)]
pub struct KnowledgeLayerSummary {
    pub notes: usize,
    pub decisions: usize,
    pub stale_count: usize,
    pub types_distribution: HashMap<String, usize>,
}

/// Fabric layer metrics
#[derive(Debug, Clone, Serialize)]
pub struct FabricLayerSummary {
    pub co_changed_pairs: usize,
}

/// Neural layer metrics
#[derive(Debug, Clone, Serialize)]
pub struct NeuralLayerSummary {
    pub active_synapses: i64,
    pub avg_energy: f64,
    pub weak_synapses_ratio: f64,
    pub dead_notes_count: i64,
}

/// Skills layer metrics
#[derive(Debug, Clone, Serialize)]
pub struct SkillsLayerSummary {
    pub total: usize,
    pub active: usize,
    pub emerging: usize,
    pub avg_cohesion: f64,
    pub total_activations: i64,
}

/// Behavioral layer metrics (Pattern Federation)
#[derive(Debug, Clone, Serialize)]
pub struct BehavioralLayerSummary {
    pub protocols: usize,
    pub states: usize,
    pub transitions: usize,
    pub system_protocols: usize,
    pub business_protocols: usize,
    pub skill_linked: usize,
}

/// Full intelligence summary response
#[derive(Debug, Clone, Serialize)]
pub struct IntelligenceSummaryResponse {
    pub code: CodeLayerSummary,
    pub knowledge: KnowledgeLayerSummary,
    pub fabric: FabricLayerSummary,
    pub neural: NeuralLayerSummary,
    pub skills: SkillsLayerSummary,
    pub behavioral: BehavioralLayerSummary,
}

// ============================================================================
// Workspace-specific types
// ============================================================================

/// Metadata about a project in the workspace graph
#[derive(Debug, Clone, Serialize)]
pub struct ProjectGraphMeta {
    pub id: String,
    pub name: String,
    pub slug: String,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Response for GET /api/workspaces/:slug/graph
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceGraphResponse {
    pub projects: Vec<ProjectGraphMeta>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub communities: Vec<GraphCommunity>,
    pub stats: HashMap<String, LayerStats>,
    /// Future: cross-project edges (shared dependencies, etc.)
    pub cross_project_edges: Vec<GraphEdge>,
}

/// Per-project intelligence summary within a workspace
#[derive(Debug, Clone, Serialize)]
pub struct ProjectIntelligenceSummary {
    pub project_id: String,
    pub project_name: String,
    pub project_slug: String,
    pub summary: IntelligenceSummaryResponse,
}

/// Response for GET /api/workspaces/:slug/intelligence/summary
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceIntelligenceSummaryResponse {
    pub aggregated: IntelligenceSummaryResponse,
    pub per_project: Vec<ProjectIntelligenceSummary>,
}

// ============================================================================
// Reusable graph-building helpers
// ============================================================================

/// Build multi-layer graph data for a single project.
///
/// Returns (nodes, edges, communities, stats) — reused by both project and workspace handlers.
pub async fn build_project_graph_data(
    neo4j: &dyn GraphStore,
    project: &ProjectNode,
    requested_layers: &[String],
    limit: usize,
    community_filter: Option<i64>,
) -> Result<
    (
        Vec<GraphNode>,
        Vec<GraphEdge>,
        Vec<GraphCommunity>,
        HashMap<String, LayerStats>,
    ),
    AppError,
> {
    let mut all_nodes = Vec::new();
    let mut all_edges = Vec::new();
    let mut communities = Vec::new();
    let mut stats = HashMap::new();

    // Set of node IDs present in the graph (for edge filtering)
    let mut node_ids: HashSet<String> = HashSet::new();

    // ---- Code Layer ----
    if requested_layers.contains(&"code".to_string()) {
        // Get files with GDS analytics (pagerank, betweenness, community_id)
        let connected_files = neo4j
            .get_most_connected_files_for_project(project.id, limit)
            .await
            .unwrap_or_default();

        // Build analytics lookup by path
        let analytics_map: HashMap<&str, _> = connected_files
            .iter()
            .map(|f| (f.path.as_str(), f))
            .collect();

        // Get ALL files for the project (connected_files may miss orphans)
        let all_files = neo4j
            .list_project_files(project.id)
            .await
            .unwrap_or_default();

        // Build code nodes
        let mut code_node_count = 0usize;
        for file in &all_files {
            let analytics = analytics_map.get(file.path.as_str());
            let community_id = analytics.and_then(|a| a.community_id);

            // Apply community filter if specified
            if let Some(filter_community) = community_filter {
                if community_id != Some(filter_community) {
                    continue;
                }
            }

            // Respect limit
            if code_node_count >= limit {
                break;
            }

            let label = file.path.rsplit('/').next().unwrap_or(&file.path);

            all_nodes.push(GraphNode {
                id: file.path.clone(),
                node_type: "file".to_string(),
                label: label.to_string(),
                layer: "code".to_string(),
                attributes: Some(serde_json::json!({
                    "language": file.language,
                    "pagerank": analytics.and_then(|a| a.pagerank),
                    "betweenness": analytics.and_then(|a| a.betweenness),
                    "community_id": community_id,
                    "community_label": analytics.and_then(|a| a.community_label.clone()),
                    "imports": analytics.map(|a| a.imports).unwrap_or(0),
                    "dependents": analytics.map(|a| a.dependents).unwrap_or(0),
                })),
            });
            node_ids.insert(file.path.clone());
            code_node_count += 1;
        }

        // Get IMPORTS edges (file-level)
        let import_edges = neo4j
            .get_project_import_edges(project.id)
            .await
            .unwrap_or_default();

        let mut code_edge_count = 0usize;
        for (source, target) in &import_edges {
            if node_ids.contains(source.as_str()) && node_ids.contains(target.as_str()) {
                all_edges.push(GraphEdge {
                    source: source.clone(),
                    target: target.clone(),
                    edge_type: "IMPORTS".to_string(),
                    layer: "code".to_string(),
                    attributes: None,
                });
                code_edge_count += 1;
            }
        }

        // Get sub-file symbols: Function, Struct, Trait, Enum
        let symbols = neo4j
            .list_project_symbols(project.id, limit)
            .await
            .unwrap_or_default();

        for (sym_id, sym_name, sym_type, file_path, visibility, line_start) in &symbols {
            if code_node_count >= limit {
                break;
            }
            all_nodes.push(GraphNode {
                id: sym_id.clone(),
                node_type: sym_type.clone(),
                label: sym_name.clone(),
                layer: "code".to_string(),
                attributes: Some(serde_json::json!({
                    "file_path": file_path,
                    "visibility": visibility,
                    "line_start": line_start,
                })),
            });
            node_ids.insert(sym_id.clone());
            code_node_count += 1;

            // CONTAINS edge: file → symbol
            if node_ids.contains(file_path) {
                all_edges.push(GraphEdge {
                    source: file_path.clone(),
                    target: sym_id.clone(),
                    edge_type: "CONTAINS".to_string(),
                    layer: "code".to_string(),
                    attributes: None,
                });
                code_edge_count += 1;
            }
        }

        // CALLS edges between functions
        let call_edges = neo4j
            .get_project_call_edges(project.id)
            .await
            .unwrap_or_default();

        for (caller_id, callee_id) in &call_edges {
            if node_ids.contains(caller_id.as_str()) && node_ids.contains(callee_id.as_str()) {
                all_edges.push(GraphEdge {
                    source: caller_id.clone(),
                    target: callee_id.clone(),
                    edge_type: "CALLS".to_string(),
                    layer: "code".to_string(),
                    attributes: None,
                });
                code_edge_count += 1;
            }
        }

        // EXTENDS / IMPLEMENTS edges between structs/traits
        let inheritance_edges = neo4j
            .get_project_inheritance_edges(project.id)
            .await
            .unwrap_or_default();

        for (source_id, target_id, rel_type) in &inheritance_edges {
            if node_ids.contains(source_id.as_str()) && node_ids.contains(target_id.as_str()) {
                all_edges.push(GraphEdge {
                    source: source_id.clone(),
                    target: target_id.clone(),
                    edge_type: rel_type.clone(),
                    layer: "code".to_string(),
                    attributes: None,
                });
                code_edge_count += 1;
            }
        }

        stats.insert(
            "code".to_string(),
            LayerStats {
                nodes: code_node_count,
                edges: code_edge_count,
            },
        );

        // Communities (from GDS Louvain clustering)
        if let Ok(comms) = neo4j.get_project_communities(project.id).await {
            communities = comms
                .into_iter()
                .map(|c| GraphCommunity {
                    id: c.community_id,
                    label: c.community_label,
                    file_count: c.file_count,
                    key_files: c.key_files,
                })
                .collect();
        }
    }

    // ---- Feature Graphs (overlay in code layer — groups of code entities) ----
    if requested_layers.contains(&"code".to_string()) {
        let feature_graphs = neo4j
            .list_feature_graphs(Some(project.id))
            .await
            .unwrap_or_default();

        // Fire all get_feature_graph_detail calls concurrently
        let detail_futures: Vec<_> = feature_graphs
            .iter()
            .map(|fg| neo4j.get_feature_graph_detail(fg.id))
            .collect();
        let details = futures::future::join_all(detail_futures).await;

        for (fg, detail_res) in feature_graphs.iter().zip(details) {
            all_nodes.push(GraphNode {
                id: fg.id.to_string(),
                node_type: "feature_graph".to_string(),
                label: fg.name.clone(),
                layer: "code".to_string(),
                attributes: Some(serde_json::json!({
                    "description": fg.description,
                    "entity_count": fg.entity_count,
                    "entry_function": fg.entry_function,
                })),
            });
            node_ids.insert(fg.id.to_string());

            // INCLUDES_ENTITY edges (feature_graph → code entity)
            if let Ok(Some(detail)) = detail_res {
                for entity in &detail.entities {
                    if node_ids.contains(&entity.entity_id) {
                        all_edges.push(GraphEdge {
                            source: fg.id.to_string(),
                            target: entity.entity_id.clone(),
                            edge_type: "INCLUDES_ENTITY".to_string(),
                            layer: "code".to_string(),
                            attributes: Some(serde_json::json!({
                                "role": entity.role,
                            })),
                        });
                    }
                }
            }
        }
    }

    // ---- Fabric Layer (CO_CHANGED edges between files) ----
    if requested_layers.contains(&"fabric".to_string()) {
        let co_changes = neo4j
            .get_co_change_graph(project.id, 1, limit as i64)
            .await
            .unwrap_or_default();

        let mut fabric_edge_count = 0usize;
        for pair in &co_changes {
            all_edges.push(GraphEdge {
                source: pair.file_a.clone(),
                target: pair.file_b.clone(),
                edge_type: "CO_CHANGED".to_string(),
                layer: "fabric".to_string(),
                attributes: Some(serde_json::json!({
                    "co_change_count": pair.count,
                    "last_at": pair.last_at,
                })),
            });
            fabric_edge_count += 1;
        }

        stats.insert(
            "fabric".to_string(),
            LayerStats {
                nodes: 0,
                edges: fabric_edge_count,
            },
        );
    }

    // ---- Knowledge Layer (Notes + Decisions + LINKED_TO + AFFECTS edges) ----
    if requested_layers.contains(&"knowledge".to_string()) {
        let mut knowledge_node_count = 0usize;
        let mut knowledge_edge_count = 0usize;

        // Get all project notes (active only)
        let note_filters = crate::notes::NoteFilters {
            status: Some(vec![crate::notes::NoteStatus::Active]),
            ..Default::default()
        };
        let (notes, _) = neo4j
            .list_notes(Some(project.id), None, &note_filters)
            .await
            .unwrap_or_default();

        for note in &notes {
            if knowledge_node_count >= limit {
                break;
            }
            let label = if note.content.chars().count() > 40 {
                let end = note
                    .content
                    .char_indices()
                    .nth(40)
                    .map(|(i, _)| i)
                    .unwrap_or(note.content.len());
                format!("{}…", &note.content[..end])
            } else {
                note.content.clone()
            };
            all_nodes.push(GraphNode {
                id: note.id.to_string(),
                node_type: "note".to_string(),
                label,
                layer: "knowledge".to_string(),
                attributes: Some(serde_json::json!({
                    "note_type": format!("{:?}", note.note_type).to_lowercase(),
                    "importance": format!("{:?}", note.importance).to_lowercase(),
                    "energy": note.energy,
                    "status": format!("{:?}", note.status).to_lowercase(),
                })),
            });
            node_ids.insert(note.id.to_string());
            knowledge_node_count += 1;
        }

        // Get decisions for this project
        let decisions_with_affects = neo4j
            .get_project_decisions_for_graph(project.id)
            .await
            .unwrap_or_default();

        for (decision, affects) in &decisions_with_affects {
            if knowledge_node_count >= limit {
                break;
            }
            let label = if decision.description.chars().count() > 40 {
                let end = decision
                    .description
                    .char_indices()
                    .nth(40)
                    .map(|(i, _)| i)
                    .unwrap_or(decision.description.len());
                format!("{}…", &decision.description[..end])
            } else {
                decision.description.clone()
            };
            all_nodes.push(GraphNode {
                id: decision.id.to_string(),
                node_type: "decision".to_string(),
                label,
                layer: "knowledge".to_string(),
                attributes: Some(serde_json::json!({
                    "status": format!("{:?}", decision.status).to_lowercase(),
                    "chosen_option": decision.chosen_option,
                    "rationale": decision.rationale,
                })),
            });
            node_ids.insert(decision.id.to_string());
            knowledge_node_count += 1;

            // AFFECTS edges (decision → entity)
            for affect in affects {
                let target_id = &affect.entity_id;
                if node_ids.contains(target_id) {
                    all_edges.push(GraphEdge {
                        source: decision.id.to_string(),
                        target: target_id.clone(),
                        edge_type: "AFFECTS".to_string(),
                        layer: "knowledge".to_string(),
                        attributes: Some(serde_json::json!({
                            "entity_type": affect.entity_type,
                            "impact": affect.impact_description,
                        })),
                    });
                    knowledge_edge_count += 1;
                }
            }
        }

        // LINKED_TO edges (note → code entity)
        let note_links = neo4j
            .get_project_note_entity_links(project.id)
            .await
            .unwrap_or_default();

        for (note_id, _entity_type, entity_id) in &note_links {
            if node_ids.contains(note_id.as_str()) && node_ids.contains(entity_id.as_str()) {
                all_edges.push(GraphEdge {
                    source: note_id.clone(),
                    target: entity_id.clone(),
                    edge_type: "LINKED_TO".to_string(),
                    layer: "knowledge".to_string(),
                    attributes: None,
                });
                knowledge_edge_count += 1;
            }
        }

        // Constraints (via Project → Plan → Constraint)
        let project_constraints = neo4j
            .get_project_constraints(project.id)
            .await
            .unwrap_or_default();

        for (constraint, _plan_id) in &project_constraints {
            if knowledge_node_count >= limit {
                break;
            }
            let label = if constraint.description.chars().count() > 40 {
                let end = constraint
                    .description
                    .char_indices()
                    .nth(40)
                    .map(|(i, _)| i)
                    .unwrap_or(constraint.description.len());
                format!("{}…", &constraint.description[..end])
            } else {
                constraint.description.clone()
            };
            all_nodes.push(GraphNode {
                id: constraint.id.to_string(),
                node_type: "constraint".to_string(),
                label,
                layer: "knowledge".to_string(),
                attributes: Some(serde_json::json!({
                    "constraint_type": format!("{:?}", constraint.constraint_type).to_lowercase(),
                    "enforced_by": constraint.enforced_by,
                })),
            });
            node_ids.insert(constraint.id.to_string());
            knowledge_node_count += 1;
        }

        stats.insert(
            "knowledge".to_string(),
            LayerStats {
                nodes: knowledge_node_count,
                edges: knowledge_edge_count,
            },
        );
    }

    // ---- Neural Layer (SYNAPSE edges between notes) ----
    if requested_layers.contains(&"neural".to_string()) {
        let synapses = neo4j
            .get_project_note_synapses(project.id, 0.1)
            .await
            .unwrap_or_default();

        let mut neural_edge_count = 0usize;
        for (source, target, weight) in &synapses {
            // Only include edges where both notes are already in the graph
            if node_ids.contains(source.as_str()) && node_ids.contains(target.as_str()) {
                all_edges.push(GraphEdge {
                    source: source.clone(),
                    target: target.clone(),
                    edge_type: "SYNAPSE".to_string(),
                    layer: "neural".to_string(),
                    attributes: Some(serde_json::json!({
                        "weight": weight,
                    })),
                });
                neural_edge_count += 1;
            }
        }

        stats.insert(
            "neural".to_string(),
            LayerStats {
                nodes: 0,
                edges: neural_edge_count,
            },
        );
    }

    // ---- Skills Layer (Skill nodes + HAS_MEMBER edges to notes/decisions) ----
    if requested_layers.contains(&"skills".to_string()) {
        let mut skills_node_count = 0usize;
        let mut skills_edge_count = 0usize;

        let (skills, _) = neo4j
            .list_skills(project.id, None, limit, 0)
            .await
            .unwrap_or_default();

        for skill in &skills {
            if skills_node_count >= limit {
                break;
            }
            all_nodes.push(GraphNode {
                id: skill.id.to_string(),
                node_type: "skill".to_string(),
                label: skill.name.clone(),
                layer: "skills".to_string(),
                attributes: Some(serde_json::json!({
                    "status": format!("{:?}", skill.status).to_lowercase(),
                    "energy": skill.energy,
                    "cohesion": skill.cohesion,
                })),
            });
            node_ids.insert(skill.id.to_string());
            skills_node_count += 1;

            // HAS_MEMBER edges to notes/decisions
            if let Ok((member_notes, member_decisions)) = neo4j.get_skill_members(skill.id).await {
                for note in &member_notes {
                    if node_ids.contains(&note.id.to_string()) {
                        all_edges.push(GraphEdge {
                            source: skill.id.to_string(),
                            target: note.id.to_string(),
                            edge_type: "HAS_MEMBER".to_string(),
                            layer: "skills".to_string(),
                            attributes: Some(serde_json::json!({
                                "member_type": "note",
                            })),
                        });
                        skills_edge_count += 1;
                    }
                }
                for decision in &member_decisions {
                    if node_ids.contains(&decision.id.to_string()) {
                        all_edges.push(GraphEdge {
                            source: skill.id.to_string(),
                            target: decision.id.to_string(),
                            edge_type: "HAS_MEMBER".to_string(),
                            layer: "skills".to_string(),
                            attributes: Some(serde_json::json!({
                                "member_type": "decision",
                            })),
                        });
                        skills_edge_count += 1;
                    }
                }
            }
        }

        stats.insert(
            "skills".to_string(),
            LayerStats {
                nodes: skills_node_count,
                edges: skills_edge_count,
            },
        );
    }

    // ---- Behavioral Layer (Protocol FSMs + HAS_STATE/HAS_TRANSITION edges) ----
    if requested_layers.contains(&"behavioral".to_string()) {
        let mut behavioral_node_count = 0usize;
        let mut behavioral_edge_count = 0usize;

        let (protocols, _) = neo4j
            .list_protocols(project.id, None, limit, 0)
            .await
            .unwrap_or_default();

        for protocol in &protocols {
            if behavioral_node_count >= limit {
                break;
            }
            all_nodes.push(GraphNode {
                id: protocol.id.to_string(),
                node_type: "protocol".to_string(),
                label: protocol.name.clone(),
                layer: "behavioral".to_string(),
                attributes: Some(serde_json::json!({
                    "category": protocol.protocol_category.to_string(),
                    "description": protocol.description,
                    "skill_id": protocol.skill_id.map(|s| s.to_string()),
                    "relevance_vector": protocol.relevance_vector.as_ref().map(|rv| serde_json::json!({
                        "phase": rv.phase,
                        "structure": rv.structure,
                        "domain": rv.domain,
                        "resource": rv.resource,
                        "lifecycle": rv.lifecycle,
                    })),
                })),
            });
            node_ids.insert(protocol.id.to_string());
            behavioral_node_count += 1;

            // HAS_STATE edges (protocol → state)
            let states = neo4j
                .get_protocol_states(protocol.id)
                .await
                .unwrap_or_default();

            for state in &states {
                if behavioral_node_count >= limit {
                    break;
                }
                all_nodes.push(GraphNode {
                    id: state.id.to_string(),
                    node_type: "protocol_state".to_string(),
                    label: state.name.clone(),
                    layer: "behavioral".to_string(),
                    attributes: Some(serde_json::json!({
                        "state_type": state.state_type.to_string(),
                        "action": state.action,
                    })),
                });
                node_ids.insert(state.id.to_string());
                behavioral_node_count += 1;

                all_edges.push(GraphEdge {
                    source: protocol.id.to_string(),
                    target: state.id.to_string(),
                    edge_type: "HAS_STATE".to_string(),
                    layer: "behavioral".to_string(),
                    attributes: None,
                });
                behavioral_edge_count += 1;
            }

            // HAS_TRANSITION edges (protocol → transition, with from/to state refs)
            let transitions = neo4j
                .get_protocol_transitions(protocol.id)
                .await
                .unwrap_or_default();

            for transition in &transitions {
                // Transition edges rendered as from_state → to_state
                let from_str = transition.from_state.to_string();
                let to_str = transition.to_state.to_string();
                if node_ids.contains(&from_str) && node_ids.contains(&to_str) {
                    all_edges.push(GraphEdge {
                        source: from_str,
                        target: to_str,
                        edge_type: "TRANSITION".to_string(),
                        layer: "behavioral".to_string(),
                        attributes: Some(serde_json::json!({
                            "trigger": transition.trigger,
                            "guard": transition.guard,
                        })),
                    });
                    behavioral_edge_count += 1;
                }
            }

            // BELONGS_TO_SKILL edge (protocol → skill, if linked)
            if let Some(skill_id) = protocol.skill_id {
                let skill_str = skill_id.to_string();
                if node_ids.contains(&skill_str) {
                    all_edges.push(GraphEdge {
                        source: protocol.id.to_string(),
                        target: skill_str,
                        edge_type: "BELONGS_TO_SKILL".to_string(),
                        layer: "behavioral".to_string(),
                        attributes: None,
                    });
                    behavioral_edge_count += 1;
                }
            }
        }

        stats.insert(
            "behavioral".to_string(),
            LayerStats {
                nodes: behavioral_node_count,
                edges: behavioral_edge_count,
            },
        );
    }

    Ok((all_nodes, all_edges, communities, stats))
}

/// Build intelligence summary metrics for a single project.
///
/// Queries all 6 layers in parallel via tokio::join!.
/// Reused by both project and workspace handlers.
pub async fn build_intelligence_summary(
    neo4j: &dyn GraphStore,
    project_id: Uuid,
) -> Result<IntelligenceSummaryResponse, AppError> {
    let pid = project_id;

    // Build filters outside tokio::join! to avoid temporary lifetime issues
    let note_filters = crate::notes::NoteFilters {
        limit: Some(0),
        ..Default::default()
    };

    // Fire all queries in parallel
    let (
        file_count_res,
        lang_stats_res,
        communities_res,
        hotspots_res,
        health_res,
        notes_res,
        stale_res,
        co_change_res,
        neural_res,
        skills_all_res,
        protocols_res,
    ) = tokio::join!(
        neo4j.count_project_files(pid),
        neo4j.get_language_stats_for_project(pid),
        neo4j.get_project_communities(pid),
        neo4j.get_top_hotspots(pid, 5),
        neo4j.get_code_health_report(pid, 200),
        neo4j.list_notes(Some(pid), None, &note_filters),
        neo4j.get_notes_needing_review(Some(pid)),
        neo4j.get_co_change_graph(pid, 1, 100_000),
        neo4j.get_neural_metrics(pid),
        neo4j.list_skills(pid, None, 1000, 0),
        neo4j.list_protocols(pid, None, 10_000, 0),
    );

    // === Code layer ===
    let file_count = file_count_res.unwrap_or(0);
    let lang_stats = lang_stats_res.unwrap_or_default();
    let function_count: usize = lang_stats.iter().map(|l| l.file_count).sum();
    let communities_list = communities_res.unwrap_or_default();
    let hotspots = hotspots_res.unwrap_or_default();
    let health = health_res.ok();

    let code = CodeLayerSummary {
        files: file_count,
        functions: function_count,
        communities: communities_list.len(),
        hotspots: hotspots
            .iter()
            .map(|h| HotspotEntry {
                path: h.path.clone(),
                churn_score: h.churn_score,
            })
            .collect(),
        orphans: health.as_ref().map(|h| h.orphan_files.len()).unwrap_or(0),
    };

    // === Knowledge layer ===
    let (_, notes_total) = notes_res.unwrap_or((vec![], 0));
    let stale_notes = stale_res.unwrap_or_default();

    let mut types_distribution = HashMap::new();
    for note in &stale_notes {
        *types_distribution
            .entry(format!("{:?}", note.note_type).to_lowercase())
            .or_insert(0usize) += 1;
    }

    let knowledge = KnowledgeLayerSummary {
        notes: notes_total,
        decisions: 0,
        stale_count: stale_notes.len(),
        types_distribution,
    };

    // === Fabric layer ===
    let co_changes = co_change_res.unwrap_or_default();

    let fabric = FabricLayerSummary {
        co_changed_pairs: co_changes.len(),
    };

    // === Neural layer ===
    let neural_metrics = neural_res.ok();

    let neural = NeuralLayerSummary {
        active_synapses: neural_metrics
            .as_ref()
            .map(|m| m.active_synapses)
            .unwrap_or(0),
        avg_energy: neural_metrics.as_ref().map(|m| m.avg_energy).unwrap_or(0.0),
        weak_synapses_ratio: neural_metrics
            .as_ref()
            .map(|m| m.weak_synapses_ratio)
            .unwrap_or(0.0),
        dead_notes_count: neural_metrics
            .as_ref()
            .map(|m| m.dead_notes_count)
            .unwrap_or(0),
    };

    // === Skills layer ===
    let (all_skills, skills_total) = skills_all_res.unwrap_or((vec![], 0));
    let active_count = all_skills
        .iter()
        .filter(|s| s.status == crate::skills::SkillStatus::Active)
        .count();
    let emerging_count = all_skills
        .iter()
        .filter(|s| s.status == crate::skills::SkillStatus::Emerging)
        .count();
    let avg_cohesion = if all_skills.is_empty() {
        0.0
    } else {
        all_skills.iter().map(|s| s.cohesion).sum::<f64>() / all_skills.len() as f64
    };
    let total_activations: i64 = all_skills.iter().map(|s| s.activation_count).sum();

    let skills = SkillsLayerSummary {
        total: skills_total,
        active: active_count,
        emerging: emerging_count,
        avg_cohesion,
        total_activations,
    };

    // === Behavioral layer (Protocol Federation) ===
    let (all_protocols, _protocols_total) = protocols_res.unwrap_or((vec![], 0));

    // Count states and transitions across all protocols (fire in parallel)
    let mut total_states = 0usize;
    let mut total_transitions = 0usize;
    for proto in &all_protocols {
        let (states_res, transitions_res) = tokio::join!(
            neo4j.get_protocol_states(proto.id),
            neo4j.get_protocol_transitions(proto.id),
        );
        total_states += states_res.map(|s| s.len()).unwrap_or(0);
        total_transitions += transitions_res.map(|t| t.len()).unwrap_or(0);
    }

    let system_count = all_protocols
        .iter()
        .filter(|p| p.protocol_category == crate::protocol::ProtocolCategory::System)
        .count();
    let business_count = all_protocols
        .iter()
        .filter(|p| p.protocol_category == crate::protocol::ProtocolCategory::Business)
        .count();
    let skill_linked_count = all_protocols
        .iter()
        .filter(|p| p.skill_id.is_some())
        .count();

    let behavioral = BehavioralLayerSummary {
        protocols: all_protocols.len(),
        states: total_states,
        transitions: total_transitions,
        system_protocols: system_count,
        business_protocols: business_count,
        skill_linked: skill_linked_count,
    };

    Ok(IntelligenceSummaryResponse {
        code,
        knowledge,
        fabric,
        neural,
        skills,
        behavioral,
    })
}
