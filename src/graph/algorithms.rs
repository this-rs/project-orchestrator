//! Graph analytics algorithms.
//!
//! Implements core graph data science algorithms on petgraph graphs:
//! - **PageRank** — power iteration (custom implementation)
//! - **Betweenness centrality** — via `rustworkx_core::centrality::betweenness_centrality`
//! - **Community detection (Louvain)** — custom implementation
//! - **Clustering coefficient** — local clustering per node
//! - **Weakly connected components** — via petgraph's `algo::connected_components` on undirected view
//!
//! ## GraIL-extended algorithms (Plans 1-10):
//! - **Structural DNA** — K-anchor distance vectors for positional fingerprinting
//! - **WL Subgraph Hash** — Weisfeiler-Lehman hash for structural isomorphism
//! - **compute_all_extended** — orchestrated pipeline with timing & error resilience
//!
//! All algorithms operate on `CodeGraph` and return results indexed by node ID (String).
//! The Louvain algorithm is implemented from scratch because the `graphina` crate
//! requires Rust 1.86+ (our MSRV target is 1.70+).

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::models::{
    AnalysisProfile, AnalyticsConfig, CodeGraph, CodeHealthReport, CommunityInfo, ComponentInfo,
    ComputeAllResult, ComputeMode, GrailConfig, GrailStats, GraphAnalytics, NodeMetrics,
    RankCluster, RankConfidence, RankedList, RankedResult, StepTiming,
};

// ============================================================================
// PageRank (power iteration)
// ============================================================================

/// Compute PageRank scores for all nodes in the graph.
///
/// Uses the power iteration method with configurable damping factor,
/// tolerance, and max iterations. Returns scores normalized to sum ≈ 1.0.
pub fn pagerank(graph: &CodeGraph, config: &AnalyticsConfig) -> HashMap<String, f64> {
    let g = &graph.graph;
    let n = g.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let damping = config.pagerank_damping;
    let tolerance = config.pagerank_tolerance;
    let max_iter = config.pagerank_max_iterations;

    // Initialize all scores to 1/n
    let initial = 1.0 / n as f64;
    let mut scores: Vec<f64> = vec![initial; n];
    let mut new_scores: Vec<f64> = vec![0.0; n];

    // Precompute out-degrees for each node
    let out_degrees: Vec<usize> = g
        .node_indices()
        .map(|idx| g.neighbors_directed(idx, Direction::Outgoing).count())
        .collect();

    for _ in 0..max_iter {
        // Reset new scores to the teleportation base
        for s in new_scores.iter_mut() {
            *s = (1.0 - damping) / n as f64;
        }

        // Distribute scores along edges
        for idx in g.node_indices() {
            let i = idx.index();
            if out_degrees[i] > 0 {
                let contribution = damping * scores[i] / out_degrees[i] as f64;
                for neighbor in g.neighbors_directed(idx, Direction::Outgoing) {
                    new_scores[neighbor.index()] += contribution;
                }
            } else {
                // Dangling node: distribute evenly to all nodes
                let contribution = damping * scores[i] / n as f64;
                for s in new_scores.iter_mut() {
                    *s += contribution;
                }
            }
        }

        // Check convergence
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < tolerance {
            break;
        }
    }

    // Normalize to sum = 1.0
    let total: f64 = scores.iter().sum();
    if total > 0.0 {
        for s in scores.iter_mut() {
            *s /= total;
        }
    }

    // Map NodeIndex → node ID
    let mut result = HashMap::with_capacity(n);
    for idx in g.node_indices() {
        let node = &g[idx];
        result.insert(node.id.clone(), scores[idx.index()]);
    }
    result
}

// ============================================================================
// Betweenness Centrality (via rustworkx-core)
// ============================================================================

/// Compute betweenness centrality for all nodes.
///
/// Uses `rustworkx_core::centrality::betweenness_centrality` with normalization.
/// Returns scores in [0, 1] range.
pub fn betweenness_centrality(graph: &CodeGraph) -> HashMap<String, f64> {
    let g = &graph.graph;
    if g.node_count() == 0 {
        return HashMap::new();
    }

    let scores = rustworkx_core::centrality::betweenness_centrality(
        g, false, // include_endpoints
        true,  // normalized
        200,   // parallel_threshold (sequential for small graphs)
    );

    let mut result = HashMap::with_capacity(g.node_count());
    for idx in g.node_indices() {
        let node = &g[idx];
        let score = scores[idx.index()].unwrap_or(0.0);
        result.insert(node.id.clone(), score);
    }
    result
}

// ============================================================================
// Community Detection — Louvain (custom implementation)
// ============================================================================

/// Detect communities using the Louvain method.
///
/// Returns `(node_to_community, communities, modularity)`.
///
/// The algorithm works on an undirected view of the graph and maximizes
/// modularity through greedy local moves of nodes between communities.
///
/// When `config.large_graph` is `Some` AND the graph exceeds `max_nodes_full`,
/// the algorithm applies adaptive optimizations:
/// - **Edge filtering**: edges with weight < `min_confidence` are excluded
/// - **Degree-1 pre-assignment**: leaf nodes skip Louvain iterations
/// - **Timeout**: the loop aborts after `max_duration_ms` returning partial results
pub fn louvain_communities(
    graph: &CodeGraph,
    config: &AnalyticsConfig,
) -> (HashMap<String, u32>, Vec<CommunityInfo>, f64) {
    let g = &graph.graph;
    let n = g.node_count();
    let resolution = config.louvain_resolution;
    if n == 0 {
        return (HashMap::new(), vec![], 0.0);
    }

    // Determine if large-graph mode is active
    let large_graph_active = config
        .large_graph
        .as_ref()
        .map(|lg| n > lg.max_nodes_full)
        .unwrap_or(false);
    let lg_config = config.large_graph.as_ref();

    // Build undirected adjacency lists (much faster than HashMap<(usize,usize)>)
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut node_strengths: Vec<f64> = vec![0.0; n]; // weighted degree

    for edge in g.edge_references() {
        let s = edge.source().index();
        let t = edge.target().index();
        let w = edge.weight().weight;

        // Large-graph mode: filter low-confidence edges
        if large_graph_active {
            if let Some(lg) = lg_config {
                if w < lg.min_confidence {
                    continue;
                }
            }
        }

        // Undirected: add weight to both directions
        adj[s].push((t, w));
        adj[t].push((s, w));
        node_strengths[s] += w;
        node_strengths[t] += w;
    }

    let total_weight: f64 = node_strengths.iter().sum::<f64>() / 2.0;
    if total_weight == 0.0 {
        // No edges: each node is its own community
        let mut node_map = HashMap::new();
        let mut communities = Vec::new();
        for (i, idx) in g.node_indices().enumerate() {
            let id = g[idx].id.clone();
            node_map.insert(id.clone(), i as u32);
            communities.push(CommunityInfo {
                id: i as u32,
                size: 1,
                members: vec![id],
                label: format!("community_{}", i),
                cohesion: 1.0,
                enriched_by: Some("heuristic".to_string()),
            });
        }
        return (node_map, communities, 0.0);
    }

    // Initialize: each node in its own community
    let mut community: Vec<u32> = (0..n as u32).collect();

    // Large-graph mode: pre-assign degree-1 nodes to their sole neighbor's community
    let mut frozen: Vec<bool> = vec![false; n];
    if large_graph_active {
        if let Some(lg) = lg_config {
            if lg.skip_degree_one {
                for node_idx in 0..n {
                    if adj[node_idx].len() == 1 {
                        let neighbor = adj[node_idx][0].0;
                        community[node_idx] = community[neighbor];
                        frozen[node_idx] = true;
                    }
                }
            }
        }
    }

    // Maintain community total strength incrementally
    let mut comm_total_strength: HashMap<u32, f64> = HashMap::with_capacity(n);
    for (i, &ki) in node_strengths.iter().enumerate() {
        *comm_total_strength.entry(community[i]).or_default() += ki;
    }

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = config.louvain_max_iterations;

    // Timeout support
    let start_time = std::time::Instant::now();
    let timeout_ms = if large_graph_active {
        lg_config.map(|lg| lg.max_duration_ms).unwrap_or(u64::MAX)
    } else {
        u64::MAX // no timeout in classic mode
    };

    while improved && iterations < max_iterations {
        // Check timeout
        if start_time.elapsed().as_millis() as u64 > timeout_ms {
            tracing::warn!(
                "Louvain timeout after {}ms ({} iterations) — returning partial result",
                start_time.elapsed().as_millis(),
                iterations
            );
            break;
        }

        improved = false;
        iterations += 1;

        for node_idx in 0..n {
            // Skip frozen (degree-1 pre-assigned) nodes
            if frozen[node_idx] {
                continue;
            }

            let current_comm = community[node_idx];

            // Calculate sum of weights to each neighboring community using adjacency list
            let mut comm_weights: HashMap<u32, f64> = HashMap::new();
            for &(neighbor, w) in &adj[node_idx] {
                let neighbor_comm = community[neighbor];
                *comm_weights.entry(neighbor_comm).or_default() += w;
            }

            // Weight to current community
            let w_in_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);

            let ki = node_strengths[node_idx];
            let m2 = 2.0 * total_weight;

            // Try removing node from current community
            let sigma_tot_current = comm_total_strength
                .get(&current_comm)
                .copied()
                .unwrap_or(0.0);
            let remove_cost =
                w_in_current / m2 - resolution * ki * (sigma_tot_current - ki) / (m2 * m2);

            // Find best community to move to
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;

            for (&target_comm, &w_to_target) in &comm_weights {
                if target_comm == current_comm {
                    continue;
                }
                let sigma_tot_target = comm_total_strength
                    .get(&target_comm)
                    .copied()
                    .unwrap_or(0.0);
                let insert_cost = w_to_target / m2 - resolution * ki * sigma_tot_target / (m2 * m2);
                let gain = insert_cost - remove_cost;

                if gain > best_gain {
                    best_gain = gain;
                    best_comm = target_comm;
                }
            }

            if best_comm != current_comm {
                // Update comm_total_strength incrementally
                *comm_total_strength.entry(current_comm).or_default() -= ki;
                *comm_total_strength.entry(best_comm).or_default() += ki;
                community[node_idx] = best_comm;
                improved = true;
            }
        }
    }

    // Renumber communities to be contiguous (0, 1, 2, ...)
    let mut comm_remap: HashMap<u32, u32> = HashMap::new();
    let mut next_id = 0u32;
    for c in &community {
        comm_remap.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
    }
    for c in community.iter_mut() {
        *c = comm_remap[c];
    }

    // Build result maps
    let mut node_map = HashMap::with_capacity(n);
    let mut comm_members: HashMap<u32, Vec<String>> = HashMap::new();

    for idx in g.node_indices() {
        let id = g[idx].id.clone();
        let comm_id = community[idx.index()];
        node_map.insert(id.clone(), comm_id);
        comm_members.entry(comm_id).or_default().push(id);
    }

    // Build CommunityInfo with auto-generated labels (cohesion computed separately)
    let mut communities: Vec<CommunityInfo> = comm_members
        .into_iter()
        .map(|(id, members)| {
            let label = generate_community_label(&members);
            CommunityInfo {
                id,
                size: members.len(),
                members,
                label,
                cohesion: 0.0, // Computed later by compute_cohesion()
                enriched_by: Some("heuristic".to_string()),
            }
        })
        .collect();
    communities.sort_by_key(|c| std::cmp::Reverse(c.size));

    // Calculate modularity
    let modularity = compute_modularity(&community, &adj, &node_strengths, total_weight);

    (node_map, communities, modularity)
}

/// Generate a human-readable label for a community from its member IDs.
///
/// Strategy: find the longest common path prefix among File-like members.
fn generate_community_label(members: &[String]) -> String {
    if members.is_empty() {
        return "empty".to_string();
    }
    if members.len() == 1 {
        return members[0]
            .rsplit('/')
            .next()
            .unwrap_or(&members[0])
            .to_string();
    }

    // Collect path components for members that look like file paths
    let path_members: Vec<&str> = members
        .iter()
        .filter(|m| m.contains('/'))
        .map(|m| m.as_str())
        .collect();

    if path_members.is_empty() {
        // All function names — take the most common prefix
        return format!("group_{}", members.len());
    }

    // Find longest common directory prefix
    let first_parts: Vec<&str> = path_members[0].split('/').collect();
    let mut common_depth = first_parts.len().saturating_sub(1); // exclude filename

    for path in &path_members[1..] {
        let parts: Vec<&str> = path.split('/').collect();
        let mut depth = 0;
        for (a, b) in first_parts.iter().zip(parts.iter()) {
            if a == b {
                depth += 1;
            } else {
                break;
            }
        }
        common_depth = common_depth.min(depth);
    }

    if common_depth > 0 {
        let prefix = first_parts[..common_depth].join("/");
        // Return the last meaningful directory component
        prefix.rsplit('/').next().unwrap_or(&prefix).to_string()
    } else {
        format!("group_{}", members.len())
    }
}

/// Compute Newman's modularity Q for a given community assignment.
fn compute_modularity(
    community: &[u32],
    adj: &[Vec<(usize, f64)>],
    node_strengths: &[f64],
    total_weight: f64,
) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }
    let m2 = 2.0 * total_weight;
    let mut q = 0.0;

    for (i, neighbors) in adj.iter().enumerate() {
        for &(j, w) in neighbors {
            if community[i] == community[j] {
                q += w - node_strengths[i] * node_strengths[j] / m2;
            }
        }
    }
    // Each undirected edge is counted twice in the adjacency list,
    // so dividing by m2 gives the correct Q
    q / m2
}

// ============================================================================
// Cohesion Scoring — internal vs external edge ratio per community
// ============================================================================

/// Compute cohesion for each community.
///
/// Cohesion = internal_edges / (internal_edges + external_edges)
/// where:
/// - internal_edges = edges with both endpoints in the same community
/// - external_edges = edges with one endpoint in the community and one outside
///
/// For large communities (> `sample_threshold` members), we sample a subset
/// of members to estimate cohesion efficiently.
///
/// Returns a map from community ID to cohesion score (0.0–1.0).
pub fn compute_cohesion(
    graph: &CodeGraph,
    communities: &[CommunityInfo],
    node_to_community: &HashMap<String, u32>,
) -> HashMap<u32, f64> {
    let g = &graph.graph;
    let sample_threshold: usize = 50;
    let mut result = HashMap::with_capacity(communities.len());

    for community in communities {
        if community.size == 0 {
            result.insert(community.id, 1.0);
            continue;
        }
        if community.size == 1 {
            // Single-node community: no internal or external edges possible
            // by convention, treat as perfectly cohesive
            result.insert(community.id, 1.0);
            continue;
        }

        // Determine which members to inspect
        let members_to_scan: Vec<&str> = if community.members.len() > sample_threshold {
            // Deterministic sampling: take every Nth member
            let step = community.members.len() / sample_threshold;
            community
                .members
                .iter()
                .step_by(step.max(1))
                .take(sample_threshold)
                .map(|s| s.as_str())
                .collect()
        } else {
            community.members.iter().map(|s| s.as_str()).collect()
        };

        let mut internal = 0u64;
        let mut external = 0u64;

        for member_id in &members_to_scan {
            if let Some(&idx) = graph.id_to_index.get(*member_id) {
                // Check outgoing edges
                for neighbor in g.neighbors_directed(idx, Direction::Outgoing) {
                    let neighbor_id = &g[neighbor].id;
                    let neighbor_comm = node_to_community.get(neighbor_id).copied();
                    if neighbor_comm == Some(community.id) {
                        internal += 1;
                    } else {
                        external += 1;
                    }
                }
                // Check incoming edges
                for neighbor in g.neighbors_directed(idx, Direction::Incoming) {
                    let neighbor_id = &g[neighbor].id;
                    let neighbor_comm = node_to_community.get(neighbor_id).copied();
                    if neighbor_comm == Some(community.id) {
                        internal += 1;
                    } else {
                        external += 1;
                    }
                }
            }
        }

        let total = internal + external;
        let cohesion = if total == 0 {
            1.0 // No edges at all → treat as perfectly cohesive
        } else {
            internal as f64 / total as f64
        };
        result.insert(community.id, cohesion);
    }

    result
}

// ============================================================================
// Clustering Coefficient
// ============================================================================

/// Compute the local clustering coefficient for each node.
///
/// For directed graphs, we consider the undirected neighborhood:
/// coefficient = triangles / possible_triangles, where
/// possible_triangles = k * (k-1) / 2 for k = number of unique neighbors.
pub fn clustering_coefficient(graph: &CodeGraph) -> HashMap<String, f64> {
    let g = &graph.graph;
    let mut result = HashMap::with_capacity(g.node_count());

    for idx in g.node_indices() {
        let node = &g[idx];

        // Collect unique neighbors (both directions for undirected view)
        let mut neighbors: Vec<NodeIndex> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for n in g.neighbors_directed(idx, Direction::Outgoing) {
            if n != idx && seen.insert(n) {
                neighbors.push(n);
            }
        }
        for n in g.neighbors_directed(idx, Direction::Incoming) {
            if n != idx && seen.insert(n) {
                neighbors.push(n);
            }
        }

        let k = neighbors.len();
        if k < 2 {
            result.insert(node.id.clone(), 0.0);
            continue;
        }

        // Count triangles: pairs of neighbors that are connected
        let mut triangles = 0usize;

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let ni = neighbors[i];
                let nj = neighbors[j];
                // Check if ni and nj are connected (any direction)
                if g.contains_edge(ni, nj) || g.contains_edge(nj, ni) {
                    triangles += 1;
                }
            }
        }

        let possible = k * (k - 1) / 2;
        let coeff = if possible > 0 {
            triangles as f64 / possible as f64
        } else {
            0.0
        };
        result.insert(node.id.clone(), coeff);
    }

    result
}

// ============================================================================
// Weakly Connected Components
// ============================================================================

/// Identify weakly connected components (treating edges as undirected).
///
/// Returns `(node_to_component, component_infos)`.
pub fn connected_components(graph: &CodeGraph) -> (HashMap<String, u32>, Vec<ComponentInfo>) {
    let g = &graph.graph;
    let n = g.node_count();
    if n == 0 {
        return (HashMap::new(), vec![]);
    }

    // Build component assignments using BFS on undirected view
    let mut component_of: Vec<Option<u32>> = vec![None; n];
    let mut component_id = 0u32;

    for start in g.node_indices() {
        if component_of[start.index()].is_some() {
            continue;
        }
        // BFS from this node on undirected graph
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        component_of[start.index()] = Some(component_id);

        while let Some(current) = queue.pop_front() {
            // Visit neighbors in both directions
            for neighbor in g.neighbors_directed(current, Direction::Outgoing) {
                if component_of[neighbor.index()].is_none() {
                    component_of[neighbor.index()] = Some(component_id);
                    queue.push_back(neighbor);
                }
            }
            for neighbor in g.neighbors_directed(current, Direction::Incoming) {
                if component_of[neighbor.index()].is_none() {
                    component_of[neighbor.index()] = Some(component_id);
                    queue.push_back(neighbor);
                }
            }
        }
        component_id += 1;
    }

    // Build result
    let mut node_map = HashMap::with_capacity(n);
    let mut comp_members: HashMap<u32, Vec<String>> = HashMap::new();

    for idx in g.node_indices() {
        let id = g[idx].id.clone();
        let comp = component_of[idx.index()].unwrap_or(0);
        node_map.insert(id.clone(), comp);
        comp_members.entry(comp).or_default().push(id);
    }

    // Find the main (largest) component
    let max_size = comp_members.values().map(|v| v.len()).max().unwrap_or(0);

    let mut components: Vec<ComponentInfo> = comp_members
        .into_iter()
        .map(|(id, members)| ComponentInfo {
            id,
            size: members.len(),
            is_main: members.len() == max_size,
            members,
        })
        .collect();
    components.sort_by_key(|c| std::cmp::Reverse(c.size));

    (node_map, components)
}

// ============================================================================
// Code Health Report
// ============================================================================

/// Compute code health metrics from the graph and node metrics.
pub fn compute_health(
    graph: &CodeGraph,
    metrics: &HashMap<String, NodeMetrics>,
    config: &AnalyticsConfig,
) -> CodeHealthReport {
    let g = &graph.graph;

    // --- God functions: total degree above the configured percentile ---
    let mut degrees: Vec<(String, usize)> = metrics
        .iter()
        .map(|(id, m)| (id.clone(), m.in_degree + m.out_degree))
        .collect();
    degrees.sort_by_key(|(_, d)| std::cmp::Reverse(*d));

    let god_functions = if !degrees.is_empty() {
        let threshold_idx =
            ((config.god_function_percentile * degrees.len() as f64) as usize).min(degrees.len());
        let threshold_degree = if threshold_idx < degrees.len() {
            degrees[threshold_idx].1
        } else {
            usize::MAX
        };
        degrees
            .iter()
            .filter(|(_, d)| *d >= threshold_degree && *d > 0)
            .map(|(id, _)| id.clone())
            .collect()
    } else {
        vec![]
    };

    // --- Circular dependencies: SCC of size > 1 via Tarjan ---
    let sccs = petgraph::algo::tarjan_scc(&g);
    let circular_dependencies: Vec<Vec<String>> = sccs
        .into_iter()
        .filter(|scc| scc.len() > 1)
        .map(|scc| scc.into_iter().map(|idx| g[idx].id.clone()).collect())
        .collect();

    // --- Orphan files: File nodes with degree = 0 ---
    let orphan_files: Vec<String> = g
        .node_indices()
        .filter(|&idx| {
            let node = &g[idx];
            node.node_type == super::models::CodeNodeType::File
                && g.neighbors_directed(idx, Direction::Outgoing).count() == 0
                && g.neighbors_directed(idx, Direction::Incoming).count() == 0
        })
        .map(|idx| g[idx].id.clone())
        .collect();

    // --- Coupling scores ---
    let clustering_vals: Vec<f64> = metrics.values().map(|m| m.clustering_coefficient).collect();
    let avg_coupling = if !clustering_vals.is_empty() {
        clustering_vals.iter().sum::<f64>() / clustering_vals.len() as f64
    } else {
        0.0
    };
    let max_coupling = clustering_vals.iter().copied().fold(0.0f64, f64::max);

    CodeHealthReport {
        god_functions,
        circular_dependencies,
        orphan_files,
        avg_coupling,
        max_coupling,
    }
}

// ============================================================================
// Orchestrator: compute_all
// ============================================================================

/// Run all base algorithms and assemble a complete `GraphAnalytics` result.
///
/// Execution order:
/// 1. PageRank (depends on graph structure only)
/// 2. Betweenness centrality
/// 3. Community detection (Louvain)
/// 4. Clustering coefficient
/// 5. Connected components
/// 6. Health report (depends on merged metrics)
pub fn compute_all(graph: &CodeGraph, config: &AnalyticsConfig) -> GraphAnalytics {
    let start = std::time::Instant::now();

    // 1. PageRank
    let pr = pagerank(graph, config);

    // 2. Betweenness centrality
    let bc = betweenness_centrality(graph);

    // 3. Louvain communities
    let (comm_map, mut communities, modularity) = louvain_communities(graph, config);

    // 3b. Compute cohesion for each community
    let cohesion_map = compute_cohesion(graph, &communities, &comm_map);
    for community in &mut communities {
        if let Some(&coh) = cohesion_map.get(&community.id) {
            community.cohesion = coh;
        }
    }

    // 4. Clustering coefficient
    let cc = clustering_coefficient(graph);

    // 5. Connected components
    let (comp_map, components) = connected_components(graph);

    // 6. Compute in/out degrees
    let g = &graph.graph;
    let mut in_degrees: HashMap<String, usize> = HashMap::new();
    let mut out_degrees: HashMap<String, usize> = HashMap::new();
    for idx in g.node_indices() {
        let id = g[idx].id.clone();
        in_degrees.insert(
            id.clone(),
            g.neighbors_directed(idx, Direction::Incoming).count(),
        );
        out_degrees.insert(id, g.neighbors_directed(idx, Direction::Outgoing).count());
    }

    // 7. Assemble NodeMetrics per node
    let mut metrics: HashMap<String, NodeMetrics> = HashMap::with_capacity(g.node_count());
    for idx in g.node_indices() {
        let id = g[idx].id.clone();
        metrics.insert(
            id.clone(),
            NodeMetrics {
                pagerank: pr.get(&id).copied().unwrap_or(0.0),
                betweenness: bc.get(&id).copied().unwrap_or(0.0),
                community_id: comm_map.get(&id).copied().unwrap_or(0),
                clustering_coefficient: cc.get(&id).copied().unwrap_or(0.0),
                component_id: comp_map.get(&id).copied().unwrap_or(0),
                in_degree: in_degrees.get(&id).copied().unwrap_or(0),
                out_degree: out_degrees.get(&id).copied().unwrap_or(0),
            },
        );
    }

    // 8. Code health
    let health = compute_health(graph, &metrics, config);

    let elapsed = start.elapsed();

    GraphAnalytics {
        metrics,
        communities,
        components,
        health,
        modularity,
        node_count: g.node_count(),
        edge_count: g.edge_count(),
        computation_ms: elapsed.as_millis() as u64,
        profile_name: None,
    }
}

// ============================================================================
// GraIL extended pipeline: compute_all_extended
// ============================================================================

/// Helper: time a step and collect its result.
fn timed_step<T, F: FnOnce() -> Result<T, String>>(name: &str, f: F) -> (Option<T>, StepTiming) {
    let start = std::time::Instant::now();
    match f() {
        Ok(result) => {
            let elapsed = start.elapsed();
            (
                Some(result),
                StepTiming {
                    name: name.to_string(),
                    duration_ms: elapsed.as_millis() as u64,
                    success: true,
                    error: None,
                },
            )
        }
        Err(e) => {
            let elapsed = start.elapsed();
            tracing::warn!(step = name, error = %e, "GraIL pipeline step failed — dependent steps will degrade gracefully");
            (
                None,
                StepTiming {
                    name: name.to_string(),
                    duration_ms: elapsed.as_millis() as u64,
                    success: false,
                    error: Some(e),
                },
            )
        }
    }
}

/// Run the full GraIL-extended analytics pipeline.
///
/// This extends `compute_all` with additional stages from the GraIL plans:
///
/// ## Pipeline order (from note 0602a13c):
/// 1. *(Optional)* Apply profile weights → re-weight edges
/// 2. PageRank
/// 3. Betweenness centrality
/// 4. Louvain communities
/// 5. Structural DNA (K-anchor distances, depends on PageRank)
/// 6. WL subgraph hashing (independent, runs after DNA)
/// 7. Stress test top-N nodes (by PageRank)
/// 8. Missing link prediction (depends on DNA similarity)
/// 9. Context cards aggregation (depends on ALL above)
/// 10. Topology rule check (post-validation)
///
/// Each step is individually timed and error-resilient: a failing step
/// logs its error and does not block subsequent independent steps.
/// Dependent steps degrade gracefully (e.g., missing links runs with
/// 4 signals instead of 5 if DNA failed).
pub fn compute_all_extended(
    graph: &CodeGraph,
    config: &AnalyticsConfig,
    grail: &GrailConfig,
    stale_node_ids: Option<&HashSet<String>>,
) -> ComputeAllResult {
    let wall_start = std::time::Instant::now();
    let mut timings: Vec<StepTiming> = Vec::with_capacity(10);
    let mut grail_stats = GrailStats::default();

    // Determine compute mode: incremental if stale < 30% of total, else full
    let total_nodes = graph.graph.node_count();
    let (mode, incremental_targets) = match stale_node_ids {
        Some(stale) if !stale.is_empty() && total_nodes > 0 => {
            let stale_ratio = stale.len() as f64 / total_nodes as f64;
            if stale_ratio < 0.30 {
                tracing::info!(
                    stale_count = stale.len(),
                    total = total_nodes,
                    ratio = format!("{:.1}%", stale_ratio * 100.0),
                    "Incremental mode: recalculating only stale nodes + neighborhood"
                );
                (ComputeMode::Incremental, Some(stale.clone()))
            } else {
                tracing::info!(
                    stale_count = stale.len(),
                    total = total_nodes,
                    ratio = format!("{:.1}%", stale_ratio * 100.0),
                    "Full recalc: stale ratio >= 30%"
                );
                (ComputeMode::Full, None)
            }
        }
        _ => (ComputeMode::Full, None),
    };

    // --- Step 0 (optional): Apply profile weights ---
    let working_graph: CodeGraph;
    if let Some(ref profile) = grail.profile {
        let (result, timing) = timed_step("apply_profile_weights", || {
            Ok(apply_profile_weights(graph, profile))
        });
        timings.push(timing);
        working_graph = result.unwrap_or_else(|| graph.clone());
    } else {
        working_graph = graph.clone();
    }

    // --- Steps 1-8: Base analytics (PageRank, Betweenness, Louvain, Health) ---
    let (base_analytics, base_timing) =
        timed_step("base_analytics", || Ok(compute_all(&working_graph, config)));
    timings.push(base_timing);

    let analytics = base_analytics.unwrap_or_else(|| GraphAnalytics {
        metrics: HashMap::new(),
        communities: vec![],
        components: vec![],
        health: CodeHealthReport::default(),
        modularity: 0.0,
        node_count: working_graph.graph.node_count(),
        edge_count: working_graph.graph.edge_count(),
        computation_ms: 0,
        profile_name: None,
    });

    // Extract PageRank scores for downstream use
    let pagerank_scores: HashMap<String, f64> = analytics
        .metrics
        .iter()
        .map(|(id, m)| (id.clone(), m.pagerank))
        .collect();

    // --- Step 5: Structural DNA (depends on PageRank) ---
    let (dna_result, dna_timing) = timed_step("structural_dna", || {
        let mut dna = structural_dna(&working_graph, &pagerank_scores, grail.dna_k)?;
        // In incremental mode, only keep stale nodes (caller will merge with existing)
        if let Some(ref targets) = incremental_targets {
            dna.retain(|id, _| targets.contains(id));
        }
        Ok(dna)
    });
    timings.push(dna_timing);

    let structural_dna_map = dna_result.unwrap_or_default();
    grail_stats.dna_computed = structural_dna_map.len();

    // --- Step 5b: Structural Fingerprint (depends on analytics) ---
    let (fp_result, fp_timing) = timed_step("structural_fingerprint", || {
        let mut fps = compute_structural_fingerprint(&working_graph, &analytics);
        if let Some(ref targets) = incremental_targets {
            fps.retain(|id, _| targets.contains(id));
        }
        Ok(fps)
    });
    timings.push(fp_timing);

    let structural_fingerprint_map = fp_result.unwrap_or_default();
    grail_stats.fingerprints_computed = structural_fingerprint_map.len();

    // --- Step 6: WL Subgraph Hash ---
    let (wl_result, wl_timing) = timed_step("wl_subgraph_hash", || {
        let mut hashes =
            wl_subgraph_hash_all(&working_graph, grail.wl_radius, grail.wl_iterations)?;
        // In incremental mode, only keep stale nodes
        if let Some(ref targets) = incremental_targets {
            hashes.retain(|id, _| targets.contains(id));
        }
        Ok(hashes)
    });
    timings.push(wl_timing);

    let wl_hashes = wl_result.unwrap_or_default();
    grail_stats.wl_computed = wl_hashes.len();

    // --- Step 7: Stress Test top-N ---
    let (_stress_result, stress_timing) = timed_step("stress_test_top_n", || {
        // TODO(Plan 5): stress_test_top_n(&working_graph, &pagerank_scores, grail.stress_top_n)
        grail_stats.stress_tested = 0;
        Ok(())
    });
    timings.push(stress_timing);

    // --- Step 8: Missing Link Prediction (depends on DNA) ---
    if structural_dna_map.is_empty() {
        tracing::warn!(
            "missing_links will run with 4 signals instead of 5 — structural DNA unavailable"
        );
    }
    let (links_result, links_timing) = timed_step("missing_links", || {
        let co_change_data = extract_co_change_data(&working_graph);
        let dna_ref = if structural_dna_map.is_empty() {
            None
        } else {
            Some(&structural_dna_map)
        };
        let predictions = suggest_missing_links(
            &working_graph,
            &co_change_data,
            dna_ref,
            grail.missing_links_top_n,
            grail.min_plausibility,
        );
        grail_stats.links_predicted = predictions.len();
        Ok(predictions)
    });
    timings.push(links_timing);

    let predicted_links = links_result.unwrap_or_default();

    // --- Step 9: Context Cards (aggregates ALL above) ---
    let (cards_result, cards_timing) = timed_step("context_cards", || {
        let cards = compute_context_cards(graph, &analytics, &structural_dna_map, &wl_hashes, &structural_fingerprint_map);
        grail_stats.cards_computed = cards.len();
        Ok(cards)
    });
    timings.push(cards_timing);
    let context_cards = cards_result.unwrap_or_default();

    // --- Step 10: Topology Rule Check (post-validation) ---
    let (_topo_result, topo_timing) = timed_step("topology_check", || {
        // TODO(Plan 3): check_topology_rules(project_id) — requires Neo4j, not in-memory
        grail_stats.violations_found = 0;
        Ok(())
    });
    timings.push(topo_timing);

    let total_ms = wall_start.elapsed().as_millis() as u64;

    ComputeAllResult {
        analytics,
        timings,
        grail_stats,
        structural_dna: structural_dna_map,
        wl_hashes,
        structural_fingerprints: structural_fingerprint_map,
        predicted_links,
        context_cards,
        mode,
        total_ms,
    }
}

// ============================================================================
// GraIL — helper: undirected neighbors
// ============================================================================

/// Get all neighbors of a node in both directions (simulates undirected graph).
fn undirected_neighbors(
    g: &petgraph::graph::DiGraph<super::models::CodeNode, super::models::CodeEdge>,
    idx: NodeIndex,
) -> Vec<NodeIndex> {
    let mut neighbors: Vec<NodeIndex> = g
        .neighbors_directed(idx, Direction::Outgoing)
        .chain(g.neighbors_directed(idx, Direction::Incoming))
        .collect();
    neighbors.sort();
    neighbors.dedup();
    neighbors
}

// ============================================================================
// GraIL algorithms — Structural DNA (Plan 2)
// ============================================================================

/// Compute structural DNA vectors for all nodes in the graph.
///
/// DNA = vector of shortest distances from each node to K anchor nodes
/// (top-K by PageRank). Inspired by GraIL's double-radius node labeling.
///
/// Returns: HashMap<node_id, Vec<f64>> where Vec has K dimensions, normalized [0,1].
pub fn structural_dna(
    graph: &CodeGraph,
    pagerank_scores: &HashMap<String, f64>,
    k: usize,
) -> Result<HashMap<String, Vec<f64>>, String> {
    let g = &graph.graph;
    if g.node_count() == 0 {
        return Ok(HashMap::new());
    }

    // Select K anchor nodes (top-K PageRank)
    let mut scored: Vec<(&String, &f64)> = pagerank_scores.iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    let anchors: Vec<NodeIndex> = scored
        .iter()
        .take(k)
        .filter_map(|(id, _)| graph.get_index(id))
        .collect();

    if anchors.is_empty() {
        return Err("No anchor nodes found (PageRank scores empty?)".to_string());
    }

    // BFS from each anchor on undirected view → distance vectors
    let mut dna_map: HashMap<String, Vec<f64>> = HashMap::with_capacity(g.node_count());

    // Initialize all DNA vectors
    for idx in g.node_indices() {
        dna_map.insert(g[idx].id.clone(), vec![f64::MAX; anchors.len()]);
    }

    // BFS from each anchor
    for (anchor_dim, &anchor_idx) in anchors.iter().enumerate() {
        let mut visited = vec![false; g.node_count()];
        let mut queue = std::collections::VecDeque::new();
        visited[anchor_idx.index()] = true;
        queue.push_back((anchor_idx, 0u32));

        while let Some((current, dist)) = queue.pop_front() {
            let id = &g[current].id;
            if let Some(dna) = dna_map.get_mut(id) {
                dna[anchor_dim] = dist as f64;
            }

            for neighbor in undirected_neighbors(g, current) {
                if !visited[neighbor.index()] {
                    visited[neighbor.index()] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
    }

    // Normalize each dimension to [0, 1]
    let num_anchors = anchors.len();
    for dim in 0..num_anchors {
        let max_dist = dna_map
            .values()
            .map(|v| v[dim])
            .filter(|d| *d < f64::MAX)
            .fold(0.0f64, f64::max);

        if max_dist > 0.0 {
            for dna in dna_map.values_mut() {
                if dna[dim] < f64::MAX {
                    dna[dim] /= max_dist;
                } else {
                    dna[dim] = 1.0; // unreachable nodes get max distance
                }
            }
        }
    }

    Ok(dna_map)
}

// ============================================================================
// Structural Fingerprint v2 — Universal cross-project similarity
// ============================================================================

/// Convert raw values to log-percentiles for power-law distributed metrics.
///
/// Uses `log(rank+1) / log(N+1)` instead of `rank/N` to preserve discrimination
/// in the tail of power-law distributions (inspired by bibliometrics research).
fn to_log_percentiles(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let n = values.len();
    if n == 1 {
        return vec![0.5]; // single element gets median percentile
    }

    // Create (original_index, value) pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut percentiles = vec![0.0; n];
    let log_n = (n as f64 + 1.0).ln();

    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        percentiles[*orig_idx] = (rank as f64 + 1.0).ln() / log_n;
    }

    percentiles
}

/// Convert raw values to linear percentiles (rank/N).
///
/// Used for metrics with roughly uniform distributions (betweenness, type_count, etc.).
fn to_linear_percentiles(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let n = values.len();
    if n == 1 {
        return vec![0.5];
    }

    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut percentiles = vec![0.0; n];
    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        percentiles[*orig_idx] = rank as f64 / (n - 1).max(1) as f64;
    }

    percentiles
}

/// Shannon entropy of a distribution, normalized to [0, 1].
///
/// Returns 0.0 for empty/single-element distributions, 1.0 for uniform distribution.
fn normalized_shannon_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let n_categories = counts.iter().filter(|&&c| c > 0).count();
    if n_categories <= 1 {
        return 0.0;
    }
    let total_f = total as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.ln()
        })
        .sum();
    let max_entropy = (n_categories as f64).ln();
    if max_entropy == 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

/// Compute structural fingerprints for all File nodes in the graph.
///
/// Returns a 17-dimensional feature vector per file with project-independent
/// semantics, enabling cross-project comparison. See `FINGERPRINT_LABELS`
/// for the meaning of each dimension.
///
/// Uses log-percentiles for power-law metrics (degree, pagerank) and
/// linear percentiles for uniform metrics (betweenness, type_count).
///
/// Dimensions requiring Neo4j-only data (d9: avg_complexity, d10: ratio_public,
/// d11: ratio_async) are set to 0.0 and enriched later in the pipeline.
pub fn compute_structural_fingerprint(
    graph: &CodeGraph,
    analytics: &GraphAnalytics,
) -> HashMap<String, Vec<f64>> {
    use super::models::{CodeEdgeType, CodeNodeType, FINGERPRINT_DIMS};

    let g = &graph.graph;

    // Phase 1: Collect raw metrics per File node
    struct RawMetrics {
        imports_in: f64,
        imports_out: f64,
        calls_in: f64,
        calls_out: f64,
        pagerank: f64,
        betweenness: f64,
        clustering: f64,
        function_count: f64,
        type_count: f64,
        // d9 (avg_complexity), d10 (ratio_public), d11 (ratio_async) = 0.0
        // (enriched from Neo4j in pipeline step)
        fan_ratio: f64,
        co_changer_count: f64,
        community_role_raw: f64,    // will be encoded as 0.0/0.5/1.0
        neighbor_type_entropy: f64, // d15
        neighbor_degree_entropy: f64, // d16 (struc2vec-inspired)
    }

    let mut file_ids: Vec<String> = Vec::new();
    let mut raw_metrics: Vec<RawMetrics> = Vec::new();

    // Pre-compute co-change data
    let co_change_data = extract_co_change_data(graph);

    for (node_id, node_idx) in &graph.id_to_index {
        let node = &g[*node_idx];
        if node.node_type != CodeNodeType::File {
            continue;
        }

        // Get analytics metrics
        let metrics = analytics.metrics.get(node_id);
        let pagerank = metrics.map(|m| m.pagerank).unwrap_or(0.0);
        let betweenness = metrics.map(|m| m.betweenness).unwrap_or(0.0);
        let clustering = metrics.map(|m| m.clustering_coefficient).unwrap_or(0.0);
        let _community_id = metrics.map(|m| m.community_id).unwrap_or(0);

        // Count edges by type (imports/calls in/out)
        let mut imports_out = 0usize;
        let mut imports_in = 0usize;
        let mut calls_out = 0usize;
        let mut calls_in = 0usize;
        let mut function_count = 0usize;
        let mut type_count = 0usize; // structs + traits + enums

        // Outgoing edges
        for edge_ref in g.edges(*node_idx) {
            match edge_ref.weight().edge_type {
                CodeEdgeType::Imports => imports_out += 1,
                CodeEdgeType::Calls => calls_out += 1,
                CodeEdgeType::Defines => {
                    // Count children by type
                    let target = &g[edge_ref.target()];
                    match target.node_type {
                        CodeNodeType::Function => function_count += 1,
                        CodeNodeType::Struct | CodeNodeType::Trait | CodeNodeType::Enum => {
                            type_count += 1
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Incoming edges
        for edge_ref in g.edges_directed(*node_idx, petgraph::Direction::Incoming) {
            match edge_ref.weight().edge_type {
                CodeEdgeType::Imports => imports_in += 1,
                CodeEdgeType::Calls => calls_in += 1,
                _ => {}
            }
        }

        // d12: fan_ratio = imports_in / (imports_in + imports_out)
        let total_imports = imports_in + imports_out;
        let fan_ratio = if total_imports > 0 {
            imports_in as f64 / total_imports as f64
        } else {
            0.5 // neutral
        };

        // d13: co-changer count (number of files that co-change with this one)
        let co_changer_count = co_change_data
            .keys()
            .filter(|(a, b)| a == node_id || b == node_id)
            .count() as f64;

        // d14: community role — hub/bridge/peripheral (encoded later via percentiles)
        // For now, store raw total_degree for percentile-based encoding
        let total_degree = (imports_in + imports_out + calls_in + calls_out) as f64;

        // d15: neighbor type entropy — Shannon entropy over CodeNodeType of neighbors
        let mut neighbor_type_counts = [0usize; 5]; // File, Function, Struct, Trait, Enum
        for neighbor_idx in g
            .neighbors_undirected(*node_idx)
        {
            let neighbor = &g[neighbor_idx];
            match neighbor.node_type {
                CodeNodeType::File => neighbor_type_counts[0] += 1,
                CodeNodeType::Function => neighbor_type_counts[1] += 1,
                CodeNodeType::Struct => neighbor_type_counts[2] += 1,
                CodeNodeType::Trait => neighbor_type_counts[3] += 1,
                CodeNodeType::Enum => neighbor_type_counts[4] += 1,
            }
        }
        let neighbor_type_entropy = normalized_shannon_entropy(&neighbor_type_counts);

        // d16: neighbor degree entropy (struc2vec-inspired)
        // Shannon entropy of the degree distribution of neighbors
        let mut neighbor_degrees: HashMap<usize, usize> = HashMap::new();
        for neighbor_idx in g.neighbors_undirected(*node_idx) {
            let deg = g.edges(neighbor_idx).count()
                + g.edges_directed(neighbor_idx, petgraph::Direction::Incoming).count();
            *neighbor_degrees.entry(deg).or_insert(0) += 1;
        }
        let degree_counts: Vec<usize> = neighbor_degrees.values().copied().collect();
        let neighbor_degree_entropy = normalized_shannon_entropy(&degree_counts);

        file_ids.push(node_id.clone());
        raw_metrics.push(RawMetrics {
            imports_in: imports_in as f64,
            imports_out: imports_out as f64,
            calls_in: calls_in as f64,
            calls_out: calls_out as f64,
            pagerank,
            betweenness,
            clustering,
            function_count: function_count as f64,
            type_count: type_count as f64,
            fan_ratio,
            co_changer_count,
            community_role_raw: total_degree,
            neighbor_type_entropy,
            neighbor_degree_entropy,
        });
    }

    let n = file_ids.len();
    if n == 0 {
        return HashMap::new();
    }

    // Phase 2: Convert raw metrics to percentiles

    // Extract raw value vectors for each percentile-based dimension
    let imports_in_vals: Vec<f64> = raw_metrics.iter().map(|m| m.imports_in).collect();
    let imports_out_vals: Vec<f64> = raw_metrics.iter().map(|m| m.imports_out).collect();
    let calls_in_vals: Vec<f64> = raw_metrics.iter().map(|m| m.calls_in).collect();
    let calls_out_vals: Vec<f64> = raw_metrics.iter().map(|m| m.calls_out).collect();
    let pagerank_vals: Vec<f64> = raw_metrics.iter().map(|m| m.pagerank).collect();
    let betweenness_vals: Vec<f64> = raw_metrics.iter().map(|m| m.betweenness).collect();
    let function_count_vals: Vec<f64> = raw_metrics.iter().map(|m| m.function_count).collect();
    let type_count_vals: Vec<f64> = raw_metrics.iter().map(|m| m.type_count).collect();
    let co_changer_vals: Vec<f64> = raw_metrics.iter().map(|m| m.co_changer_count).collect();
    let degree_vals: Vec<f64> = raw_metrics.iter().map(|m| m.community_role_raw).collect();

    // Log-percentiles for power-law metrics
    let imports_in_pct = to_log_percentiles(&imports_in_vals);
    let imports_out_pct = to_log_percentiles(&imports_out_vals);
    let calls_in_pct = to_log_percentiles(&calls_in_vals);
    let calls_out_pct = to_log_percentiles(&calls_out_vals);
    let pagerank_pct = to_log_percentiles(&pagerank_vals);
    let function_count_pct = to_log_percentiles(&function_count_vals);

    // Linear percentiles for more uniform metrics
    let betweenness_pct = to_linear_percentiles(&betweenness_vals);
    let type_count_pct = to_linear_percentiles(&type_count_vals);
    let co_changer_pct = to_linear_percentiles(&co_changer_vals);
    let degree_pct = to_linear_percentiles(&degree_vals);
    let betweenness_raw_pct = to_linear_percentiles(&betweenness_vals);

    // Phase 3: Assemble fingerprint vectors
    let mut result = HashMap::with_capacity(n);

    for i in 0..n {
        let mut fingerprint = vec![0.0f64; FINGERPRINT_DIMS];

        // d0-d3: degree percentiles (log)
        fingerprint[0] = imports_in_pct[i];
        fingerprint[1] = imports_out_pct[i];
        fingerprint[2] = calls_in_pct[i];
        fingerprint[3] = calls_out_pct[i];

        // d4-d6: centrality
        fingerprint[4] = pagerank_pct[i];
        fingerprint[5] = betweenness_pct[i];
        fingerprint[6] = raw_metrics[i].clustering; // raw 0-1

        // d7-d8: code content (d9 = avg_complexity left at 0.0, enriched from Neo4j)
        fingerprint[7] = function_count_pct[i];
        fingerprint[8] = type_count_pct[i];
        // fingerprint[9] = 0.0; // avg_complexity_pct — enriched later

        // d10-d11: code style (left at 0.0, enriched from Neo4j)
        // fingerprint[10] = 0.0; // ratio_public — enriched later
        // fingerprint[11] = 0.0; // ratio_async — enriched later

        // d12: fan ratio (raw 0-1)
        fingerprint[12] = raw_metrics[i].fan_ratio;

        // d13: co-changer count percentile
        fingerprint[13] = co_changer_pct[i];

        // d14: community role — hub(0.0) / bridge(0.5) / peripheral(1.0)
        // Hub = top 10% by degree, bridge = top 20% by betweenness, else peripheral
        fingerprint[14] = if degree_pct[i] >= 0.9 {
            0.0 // hub
        } else if betweenness_raw_pct[i] >= 0.8 {
            0.5 // bridge
        } else {
            1.0 // peripheral
        };

        // d15: neighbor type entropy (raw 0-1)
        fingerprint[15] = raw_metrics[i].neighbor_type_entropy;

        // d16: neighbor degree entropy (raw 0-1, struc2vec-inspired)
        fingerprint[16] = raw_metrics[i].neighbor_degree_entropy;

        result.insert(file_ids[i].clone(), fingerprint);
    }

    result
}

/// Compute cosine similarity between two DNA vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Find structural twins — nodes with most similar DNA vectors.
pub fn find_structural_twins(
    dna_map: &HashMap<String, Vec<f64>>,
    target_id: &str,
    top_n: usize,
) -> Vec<(String, f64)> {
    let target_dna = match dna_map.get(target_id) {
        Some(dna) => dna,
        None => return vec![],
    };

    let mut similarities: Vec<(String, f64)> = dna_map
        .iter()
        .filter(|(id, _)| id.as_str() != target_id)
        .map(|(id, dna)| (id.clone(), cosine_similarity(target_dna, dna)))
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(top_n);
    similarities
}

// ============================================================================
// Multi-signal Structural Similarity (Fingerprint v2)
// ============================================================================

/// Fusion weights for multi-signal similarity.
const W_FINGERPRINT: f64 = 0.50;
const W_WL_HASH: f64 = 0.20;
const W_NAME: f64 = 0.20;
const W_SIZE: f64 = 0.10;

/// Compute Jaro similarity between two strings.
///
/// Returns a value in [0.0, 1.0] where 1.0 means identical.
/// This is the base for Jaro-Winkler and works well for short file-stem comparisons.
fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }
    let len1 = s1.len();
    let len2 = s2.len();
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_distance = (std::cmp::max(len1, len2) / 2).saturating_sub(1);

    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    let mut s1_matched = vec![false; len1];
    let mut s2_matched = vec![false; len2];

    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Count matching characters
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = std::cmp::min(i + match_distance + 1, len2);
        for j in start..end {
            if s2_matched[j] || s1_bytes[i] != s2_bytes[j] {
                continue;
            }
            s1_matched[i] = true;
            s2_matched[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0usize;
    for i in 0..len1 {
        if !s1_matched[i] {
            continue;
        }
        while !s2_matched[k] {
            k += 1;
        }
        if s1_bytes[i] != s2_bytes[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
}

/// Compute Jaro-Winkler similarity between two strings.
///
/// Boosts the Jaro score when the strings share a common prefix (up to 4 chars).
/// Good for file names that often share prefixes like "user_", "auth_", etc.
fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    let jaro = jaro_similarity(s1, s2);
    if jaro == 0.0 {
        return 0.0;
    }

    // Count common prefix (max 4 characters)
    let prefix_len = s1
        .chars()
        .zip(s2.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    // Winkler scaling factor p = 0.1 (standard)
    jaro + prefix_len as f64 * 0.1 * (1.0 - jaro)
}

/// Extract the file stem (name without extension and path).
///
/// "src/api/handlers.rs" → "handlers"
/// "components/UserProfile.tsx" → "UserProfile"
fn file_stem(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name)
}

/// Compute log-size similarity between two function counts.
///
/// Uses `1 - |log(a+1) - log(b+1)| / log(max+1)` to produce a value in [0, 1].
/// Files with similar function counts (scale-wise) get high scores.
fn log_size_similarity(count_a: usize, count_b: usize) -> f64 {
    let la = (count_a as f64 + 1.0).ln();
    let lb = (count_b as f64 + 1.0).ln();
    let max_log = la.max(lb);
    if max_log == 0.0 {
        return 1.0; // Both zero → identical
    }
    1.0 - (la - lb).abs() / max_log
}

/// Input data for one file in multi-signal similarity computation.
#[derive(Debug, Clone)]
pub struct FileSignals {
    /// File path (used as identifier)
    pub path: String,
    /// 17-dim structural fingerprint vector (from `compute_structural_fingerprint`)
    pub fingerprint: Vec<f64>,
    /// WL subgraph hash (from `wl_subgraph_hash_all`), None if unavailable
    pub wl_hash: Option<u64>,
    /// Number of functions defined in this file (d7 raw value)
    pub function_count: usize,
}

/// Compute multi-signal fused similarity between two files.
///
/// Combines four independent signals with fixed weights:
/// - **Fingerprint cosine** (0.50): 17-dim structural role vector comparison
/// - **WL hash match** (0.20): exact topological neighborhood match bonus
/// - **Name similarity** (0.20): Jaro-Winkler on file stems (cross-project naming patterns)
/// - **Size similarity** (0.10): log-ratio of function counts
///
/// Returns a `FingerprintSimilarity` with the fused score and per-signal breakdown.
pub fn compute_multi_signal_similarity(
    source: &FileSignals,
    target: &FileSignals,
) -> super::models::FingerprintSimilarity {
    // Signal 1: Fingerprint cosine similarity (weight 0.50)
    let fp_sim = if source.fingerprint.is_empty() || target.fingerprint.is_empty() {
        0.0
    } else {
        cosine_similarity(&source.fingerprint, &target.fingerprint)
    };

    // Signal 2: WL hash exact match (weight 0.20)
    let wl_match = match (source.wl_hash, target.wl_hash) {
        (Some(a), Some(b)) if a == b => 1.0,
        _ => 0.0,
    };

    // Signal 3: File name Jaro-Winkler similarity (weight 0.20)
    let name_sim = jaro_winkler_similarity(file_stem(&source.path), file_stem(&target.path));

    // Signal 4: Log-size similarity (weight 0.10)
    let size_sim = log_size_similarity(source.function_count, target.function_count);

    // Fused score
    let fused = W_FINGERPRINT * fp_sim + W_WL_HASH * wl_match + W_NAME * name_sim + W_SIZE * size_sim;

    super::models::FingerprintSimilarity {
        source: source.path.clone(),
        target: target.path.clone(),
        similarity: fused,
        signals: super::models::SimilaritySignals {
            fingerprint_similarity: fp_sim,
            wl_hash_match: wl_match,
            name_similarity: name_sim,
            size_similarity: size_sim,
        },
        shared_role: None, // Caller can enrich this based on community/cluster labels
    }
}

/// Find cross-project structural twins using multi-signal fusion.
///
/// Given a source file's signals and a collection of candidate files from other projects,
/// computes fused similarity for each candidate and returns the top-N most similar.
///
/// This replaces the old DNA-only cosine approach which suffered from project-specific
/// anchor bias (all similarities ≈ 0.999).
pub fn find_cross_project_twins_multi_signal(
    source: &FileSignals,
    candidates: &[FileSignals],
    top_n: usize,
) -> Vec<super::models::FingerprintSimilarity> {
    let mut results: Vec<super::models::FingerprintSimilarity> = candidates
        .iter()
        .filter(|c| c.path != source.path)
        .map(|candidate| compute_multi_signal_similarity(source, candidate))
        .collect();

    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_n);
    results
}

/// Find intra-project structural twins using multi-signal fusion.
///
/// Same as cross-project but operates within a single project's file set.
/// Uses fingerprints instead of DNA to avoid the anchor-bias problem.
pub fn find_structural_twins_multi_signal(
    source: &FileSignals,
    project_files: &[FileSignals],
    top_n: usize,
) -> Vec<super::models::FingerprintSimilarity> {
    find_cross_project_twins_multi_signal(source, project_files, top_n)
}

// ============================================================================
// Structural DNA — K-means Clustering
// ============================================================================

/// K-means clustering on structural DNA vectors.
///
/// Groups files by structural similarity into `n_clusters` clusters.
/// Each cluster represents an architectural role (e.g., handlers, models, utils).
///
/// Algorithm:
/// 1. Initialize centroids via K-means++ (spread-out initial seeds)
/// 2. Iterate assignment + update steps until convergence (max 100 iterations)
/// 3. Auto-label each cluster based on dominant file name patterns
///
/// Returns empty vec if dna_map has fewer entries than n_clusters.
pub fn cluster_dna_vectors(
    dna_map: &HashMap<String, Vec<f64>>,
    n_clusters: usize,
) -> Vec<super::models::DnaCluster> {
    if dna_map.is_empty() || n_clusters == 0 || dna_map.len() < n_clusters {
        return vec![];
    }

    let paths: Vec<&String> = dna_map.keys().collect();
    let vectors: Vec<&Vec<f64>> = paths.iter().map(|p| &dna_map[*p]).collect();
    let n = vectors.len();
    let dim = vectors[0].len();

    if dim == 0 {
        return vec![];
    }

    // --- K-means++ initialization ---
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(n_clusters);

    // First centroid: pick the first vector (deterministic for reproducibility)
    centroids.push(vectors[0].clone());

    for _ in 1..n_clusters {
        // For each point, compute min squared distance to nearest centroid
        let mut distances: Vec<f64> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| squared_euclidean(v, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Pick the point with max distance (deterministic K-means++)
        let max_idx = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.push(vectors[max_idx].clone());
        // Zero out to avoid picking same point
        distances[max_idx] = 0.0;
    }

    // --- K-means iterations ---
    let mut assignments = vec![0usize; n];
    let max_iter = 100;

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step: assign each vector to nearest centroid
        for (i, v) in vectors.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, squared_euclidean(v, c)))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0);

            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids as mean of assigned vectors
        let mut new_centroids = vec![vec![0.0; dim]; n_clusters];
        let mut counts = vec![0usize; n_clusters];

        for (i, v) in vectors.iter().enumerate() {
            let ci = assignments[i];
            counts[ci] += 1;
            for (d, val) in v.iter().enumerate() {
                new_centroids[ci][d] += val;
            }
        }

        for (ci, centroid) in new_centroids.iter_mut().enumerate().take(n_clusters) {
            if counts[ci] > 0 {
                for val in centroid.iter_mut().take(dim) {
                    *val /= counts[ci] as f64;
                }
            }
        }

        centroids = new_centroids;
    }

    // --- Build clusters ---
    let mut clusters: Vec<super::models::DnaCluster> = Vec::with_capacity(n_clusters);

    for (ci, _centroid) in centroids.iter().enumerate().take(n_clusters) {
        let member_indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &a)| a == ci)
            .map(|(i, _)| i)
            .collect();

        if member_indices.is_empty() {
            continue; // Skip empty clusters
        }

        let members: Vec<String> = member_indices.iter().map(|&i| paths[i].clone()).collect();

        // Compute intra-cluster cohesion (average pairwise cosine similarity)
        let cohesion = if members.len() <= 1 {
            1.0
        } else {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..member_indices.len() {
                for j in (i + 1)..member_indices.len() {
                    sum +=
                        cosine_similarity(vectors[member_indices[i]], vectors[member_indices[j]]);
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                1.0
            }
        };

        let label = infer_cluster_label(&members);

        clusters.push(super::models::DnaCluster {
            id: ci,
            centroid: centroids[ci].clone(),
            members,
            label,
            cohesion,
        });
    }

    // Sort by cluster size descending
    clusters.sort_by(|a, b| b.members.len().cmp(&a.members.len()));

    clusters
}

/// Squared Euclidean distance between two vectors.
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Infer a human-readable label for a cluster based on dominant file name patterns.
///
/// Looks for common suffixes/patterns like `handler`, `model`, `service`, `test`, etc.
fn infer_cluster_label(paths: &[String]) -> String {
    let patterns: &[(&str, &str)] = &[
        ("handler", "Handlers"),
        ("controller", "Controllers"),
        ("route", "Routes"),
        ("model", "Models"),
        ("schema", "Schemas"),
        ("service", "Services"),
        ("repository", "Repositories"),
        ("store", "Stores"),
        ("client", "Clients"),
        ("test", "Tests"),
        ("spec", "Tests"),
        ("mock", "Mocks"),
        ("util", "Utilities"),
        ("helper", "Helpers"),
        ("config", "Configuration"),
        ("middleware", "Middleware"),
        ("trait", "Traits"),
        ("interface", "Interfaces"),
        ("mod.rs", "Modules"),
        ("index", "Index/Entry"),
        ("lib", "Library"),
        ("main", "Entry Points"),
        ("error", "Error Handling"),
        ("types", "Types"),
        ("api", "API"),
    ];

    // Count matches for each pattern
    let mut scores: Vec<(&str, usize)> = patterns
        .iter()
        .map(|(pattern, label)| {
            let count = paths
                .iter()
                .filter(|p| {
                    let lower = p.to_lowercase();
                    let filename = lower.rsplit('/').next().unwrap_or(&lower);
                    filename.contains(pattern)
                })
                .count();
            (*label, count)
        })
        .filter(|(_, count)| *count > 0)
        .collect();

    scores.sort_by(|a, b| b.1.cmp(&a.1));

    if let Some((label, count)) = scores.first() {
        if *count * 3 >= paths.len() {
            // At least 1/3 of files match this pattern → use it
            return label.to_string();
        }
    }

    // Fallback: try to find common directory prefix
    if let Some(common_dir) = find_common_directory(paths) {
        return common_dir;
    }

    format!("Cluster ({})", paths.len())
}

/// Find the most specific common directory among paths.
fn find_common_directory(paths: &[String]) -> Option<String> {
    if paths.is_empty() {
        return None;
    }

    // Extract directory parts from each path
    let dirs: Vec<Vec<&str>> = paths
        .iter()
        .filter_map(|p| {
            let parts: Vec<&str> = p.split('/').collect();
            if parts.len() > 1 {
                Some(parts[..parts.len() - 1].to_vec())
            } else {
                None
            }
        })
        .collect();

    if dirs.is_empty() {
        return None;
    }

    // Find longest common prefix
    let first = &dirs[0];
    let mut common_len = 0;

    for i in 0..first.len() {
        if dirs.iter().all(|d| d.len() > i && d[i] == first[i]) {
            common_len = i + 1;
        } else {
            break;
        }
    }

    // Use the deepest common directory component as label
    if common_len > 0 {
        let deepest = first[common_len - 1];
        if !["src", "lib", "app", "pkg"].contains(&deepest) {
            return Some(deepest.to_string());
        }
        // If the deepest is too generic, try one level deeper if possible
        if common_len >= 2 {
            return Some(first[common_len - 2..common_len].join("/"));
        }
    }

    None
}

// ============================================================================
// GraIL algorithms — WL Subgraph Hash (Plan 7)
// ============================================================================

/// Compute Weisfeiler-Lehman hash for a single node's neighborhood.
///
/// Algorithm: BFS up to `radius` hops, then `wl_iterations` rounds of
/// WL relabeling. Final hash = hash of sorted multiset of all labels.
pub fn wl_node_hash(
    graph: &CodeGraph,
    center: NodeIndex,
    radius: usize,
    wl_iterations: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let g = &graph.graph;

    // BFS to collect neighborhood within radius (undirected view)
    let mut neighborhood: Vec<NodeIndex> = Vec::new();
    let mut visited = vec![false; g.node_count()];
    let mut queue = std::collections::VecDeque::new();
    visited[center.index()] = true;
    queue.push_back((center, 0usize));

    while let Some((node, dist)) = queue.pop_front() {
        neighborhood.push(node);
        if dist < radius {
            for neighbor in undirected_neighbors(g, node) {
                if !visited[neighbor.index()] {
                    visited[neighbor.index()] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
    }

    // Initialize labels from node type
    let mut labels: HashMap<NodeIndex, u64> = HashMap::new();
    for &idx in &neighborhood {
        let mut hasher = DefaultHasher::new();
        g[idx].node_type.hash(&mut hasher);
        labels.insert(idx, hasher.finish());
    }

    // WL iterations: relabel each node based on sorted neighbor labels
    for _ in 0..wl_iterations {
        let mut new_labels: HashMap<NodeIndex, u64> = HashMap::new();
        for &idx in &neighborhood {
            let mut neighbor_labels: Vec<u64> = undirected_neighbors(g, idx)
                .into_iter()
                .filter(|n| labels.contains_key(n))
                .map(|n| labels[&n])
                .collect();
            neighbor_labels.sort();

            let mut hasher = DefaultHasher::new();
            labels[&idx].hash(&mut hasher);
            for nl in neighbor_labels.iter() {
                nl.hash(&mut hasher);
            }
            new_labels.insert(idx, hasher.finish());
        }
        labels = new_labels;
    }

    // Final hash = hash of sorted multiset of all labels in neighborhood
    let mut all_labels: Vec<u64> = labels.values().copied().collect();
    all_labels.sort();
    let mut hasher = DefaultHasher::new();
    for label in &all_labels {
        label.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute WL hash for all nodes in the graph.
pub fn wl_subgraph_hash_all(
    graph: &CodeGraph,
    radius: usize,
    wl_iterations: usize,
) -> Result<HashMap<String, u64>, String> {
    let g = &graph.graph;
    let mut hashes: HashMap<String, u64> = HashMap::with_capacity(g.node_count());

    for idx in g.node_indices() {
        let hash = wl_node_hash(graph, idx, radius, wl_iterations);
        hashes.insert(g[idx].id.clone(), hash);
    }

    Ok(hashes)
}

/// Find groups of nodes with identical WL hash (isomorphic neighborhoods).
pub fn find_isomorphic_groups(wl_hashes: &HashMap<String, u64>) -> HashMap<u64, Vec<String>> {
    let mut groups: HashMap<u64, Vec<String>> = HashMap::new();
    for (id, hash) in wl_hashes {
        groups.entry(*hash).or_default().push(id.clone());
    }
    // Keep only groups with 2+ members
    groups.retain(|_, members| members.len() >= 2);
    groups
}

// ============================================================================
// Analysis Profiles — Contextual edge weighting (Plan 6 / GraIL R-GCN)
// ============================================================================

/// Apply an analysis profile's edge weights to a graph, returning a new weighted graph.
///
/// For each edge in the graph, the weight is multiplied by the profile's weight
/// for that edge type. Edge types not present in the profile use `default_weight`
/// (0.5 = neutral reduction).
///
/// This is O(E) — a single pass over all edges.
///
/// # Example
/// ```ignore
/// let security = profile_security();
/// let weighted = apply_profile_weights(&graph, &security);
/// let analytics = compute_all(&weighted, &config);
/// ```
pub fn apply_profile_weights(graph: &CodeGraph, profile: &AnalysisProfile) -> CodeGraph {
    let default_weight = 0.5;
    let mut weighted = graph.clone();

    for edge_idx in weighted.graph.edge_indices() {
        if let Some(edge) = weighted.graph.edge_weight_mut(edge_idx) {
            let edge_type_str = edge.edge_type.to_string();
            let profile_weight = profile
                .edge_weights
                .get(&edge_type_str)
                .copied()
                .unwrap_or(default_weight);
            edge.weight *= profile_weight;
        }
    }

    weighted
}

/// Run the full analytics pipeline on a profile-weighted graph.
///
/// 1. Applies profile edge weights via `apply_profile_weights`
/// 2. Runs `compute_all` (PageRank, Betweenness, Louvain, etc.) on the weighted graph
///
/// Returns `GraphAnalytics` computed on the weighted graph. The `profile_name` field
/// in the result indicates which profile was used.
pub fn compute_all_with_profile(
    graph: &CodeGraph,
    config: &AnalyticsConfig,
    profile: &AnalysisProfile,
) -> GraphAnalytics {
    let weighted = apply_profile_weights(graph, profile);
    let mut analytics = compute_all(&weighted, config);
    analytics.profile_name = Some(profile.name.clone());
    analytics
}

// ============================================================================
// Margin Ranking — Universal relative ranking (Plan 10 / GraIL)
// ============================================================================

/// Returns a human-readable cluster label based on cluster index (0-based).
///
/// Labels follow a severity gradient: critical → high → moderate → low → peripheral.
pub fn cluster_label(index: usize) -> String {
    match index {
        0 => "critical".to_string(),
        1 => "high".to_string(),
        2 => "moderate".to_string(),
        3 => "low".to_string(),
        _ => "peripheral".to_string(),
    }
}

/// Detect natural clusters in a sorted (descending) list of scored items.
///
/// Uses gap analysis: when the score gap between consecutive items exceeds
/// `min_gap_ratio × score_range`, a cluster boundary is created.
///
/// # Arguments
/// * `scores` — Items with their scores, **must be sorted descending by score**
/// * `min_gap_ratio` — Minimum relative gap to trigger a cluster boundary (e.g., 0.15 = 15%)
///
/// # Returns
/// A list of `RankCluster` with 1-based ranks, average scores, and labels.
pub fn detect_natural_clusters(scores: &[(String, f64)], min_gap_ratio: f64) -> Vec<RankCluster> {
    if scores.len() < 2 {
        if scores.len() == 1 {
            return vec![RankCluster {
                start_rank: 1,
                end_rank: 1,
                avg_score: scores[0].1,
                label: cluster_label(0),
            }];
        }
        return vec![];
    }

    let range = scores.first().unwrap().1 - scores.last().unwrap().1;
    if range <= 0.0 {
        // All scores are identical → single cluster
        let avg = scores.iter().map(|s| s.1).sum::<f64>() / scores.len() as f64;
        return vec![RankCluster {
            start_rank: 1,
            end_rank: scores.len(),
            avg_score: avg,
            label: cluster_label(0),
        }];
    }

    let threshold = min_gap_ratio * range;

    let mut clusters = Vec::new();
    let mut cluster_start = 0usize;

    for i in 0..scores.len() - 1 {
        let gap = scores[i].1 - scores[i + 1].1;
        if gap > threshold {
            // Close current cluster (1-based ranks)
            let slice = &scores[cluster_start..=i];
            let avg = slice.iter().map(|s| s.1).sum::<f64>() / slice.len() as f64;
            clusters.push(RankCluster {
                start_rank: cluster_start + 1,
                end_rank: i + 1,
                avg_score: avg,
                label: cluster_label(clusters.len()),
            });
            cluster_start = i + 1;
        }
    }

    // Close the last cluster
    let slice = &scores[cluster_start..];
    let avg = slice.iter().map(|s| s.1).sum::<f64>() / slice.len() as f64;
    clusters.push(RankCluster {
        start_rank: cluster_start + 1,
        end_rank: scores.len(),
        avg_score: avg,
        label: cluster_label(clusters.len()),
    });

    clusters
}

/// Transform a list of scored items into a `RankedList<T>` with margins,
/// confidence levels, and natural clusters.
///
/// Items are sorted by score descending. For each item, the margin to the
/// next and previous items is computed, and confidence is derived from the
/// minimum margin (how "safe" is this ranking position).
///
/// # Arguments
/// * `items` — Vec of (item, score) pairs
/// * `total_candidates` — Total number of candidates before any filtering
///
/// # Performance
/// O(n log n) for sort + O(n) for margins + O(n) for clusters = O(n log n) total.
pub fn into_ranked<T: Serialize + Clone>(
    mut items: Vec<(T, f64)>,
    total_candidates: usize,
) -> RankedList<T> {
    if items.is_empty() {
        return RankedList {
            items: vec![],
            total_candidates,
            score_range: (0.0, 0.0),
            natural_clusters: vec![],
        };
    }

    // Sort by score descending (stable sort to preserve input order for ties)
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let min_score = items.last().unwrap().1;
    let max_score = items.first().unwrap().1;

    // Build scored labels for cluster detection
    let scored_labels: Vec<(String, f64)> = items
        .iter()
        .enumerate()
        .map(|(i, (_, score))| (format!("item_{}", i), *score))
        .collect();

    let natural_clusters = detect_natural_clusters(&scored_labels, 0.15);

    // Build ranked results with margins and confidence
    let n = items.len();
    let ranked_items: Vec<RankedResult<T>> = items
        .into_iter()
        .enumerate()
        .map(|(i, (item, score))| {
            let margin_to_next = if i + 1 < n {
                Some(score - scored_labels[i + 1].1)
            } else {
                None
            };
            let margin_to_prev = if i > 0 {
                Some(scored_labels[i - 1].1 - score)
            } else {
                None
            };

            // Confidence = based on the minimum margin (weakest separation)
            let min_margin = match (margin_to_next, margin_to_prev) {
                (Some(mn), Some(mp)) => mn.min(mp),
                (Some(m), None) | (None, Some(m)) => m,
                (None, None) => 0.0,
            };

            RankedResult {
                item,
                rank: i + 1,
                score,
                margin_to_next,
                margin_to_prev,
                confidence: RankConfidence::from_margin(min_margin),
                signals: vec![],
            }
        })
        .collect();

    RankedList {
        items: ranked_items,
        total_candidates,
        score_range: (min_score, max_score),
        natural_clusters,
    }
}

// ============================================================================
// Bridge Subgraph — Double-radius labeling & bottleneck detection (GraIL Plan 1)
// ============================================================================

/// Compute BFS distances from a single source node to all reachable nodes
/// in the bridge subgraph. Works on an adjacency list built from raw edges.
///
/// Returns a map of `path -> distance`. Unreachable nodes are not included.
fn bfs_distances(adj: &HashMap<&str, Vec<&str>>, source: &str) -> HashMap<String, u32> {
    use std::collections::VecDeque;
    let mut distances = HashMap::new();
    distances.insert(source.to_string(), 0u32);
    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        let current_dist = distances[current];
        if let Some(neighbors) = adj.get(current) {
            for &neighbor in neighbors {
                if !distances.contains_key(neighbor) {
                    distances.insert(neighbor.to_string(), current_dist + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    distances
}

/// Compute double-radius labels for each node in the bridge subgraph.
///
/// For each node, returns `(distance_to_source, distance_to_target)`.
/// Unreachable nodes get `u32::MAX` for the unreachable direction.
///
/// ## Arguments
/// - `node_paths`: paths of all nodes in the bridge subgraph
/// - `edges`: list of `(from_path, to_path)` tuples (directed edges)
/// - `source`: source node path
/// - `target`: target node path
///
/// ## Example
/// Linear graph A→B→C with source=A, target=C:
/// - A = (0, 2), B = (1, 1), C = (2, 0)
pub fn double_radius_label(
    node_paths: &[String],
    edges: &[(String, String)],
    source: &str,
    target: &str,
) -> HashMap<String, (u32, u32)> {
    // Build undirected adjacency list for BFS traversal
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for path in node_paths {
        adj.entry(path.as_str()).or_default();
    }
    for (from, to) in edges {
        adj.entry(from.as_str()).or_default().push(to.as_str());
        adj.entry(to.as_str()).or_default().push(from.as_str());
    }

    let dist_from_source = bfs_distances(&adj, source);
    let dist_from_target = bfs_distances(&adj, target);

    let mut labels = HashMap::with_capacity(node_paths.len());
    for path in node_paths {
        let d_s = dist_from_source
            .get(path.as_str())
            .copied()
            .unwrap_or(u32::MAX);
        let d_t = dist_from_target
            .get(path.as_str())
            .copied()
            .unwrap_or(u32::MAX);
        labels.insert(path.clone(), (d_s, d_t));
    }
    labels
}

/// Find bottleneck nodes in the bridge subgraph using Brandes' betweenness
/// centrality algorithm on the local subgraph. Returns top-N node paths
/// sorted by betweenness descending.
///
/// Excludes source and target from the results (they're anchors, not bottlenecks).
///
/// ## Arguments
/// - `node_paths`: paths of all nodes in the bridge subgraph
/// - `edges`: list of `(from_path, to_path)` tuples
/// - `source`: source node path (excluded from results)
/// - `target`: target node path (excluded from results)
/// - `top_n`: number of bottleneck nodes to return
pub fn find_bottleneck_nodes(
    node_paths: &[String],
    edges: &[(String, String)],
    source: &str,
    target: &str,
    top_n: usize,
) -> Vec<String> {
    use std::collections::VecDeque;

    if node_paths.len() < 3 {
        return Vec::new(); // Need at least source + target + 1 intermediate
    }

    // Map paths to indices for efficient computation
    let path_to_idx: HashMap<&str, usize> = node_paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();
    let n = node_paths.len();

    // Build undirected adjacency list (by index)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (from, to) in edges {
        if let (Some(&fi), Some(&ti)) =
            (path_to_idx.get(from.as_str()), path_to_idx.get(to.as_str()))
        {
            if !adj[fi].contains(&ti) {
                adj[fi].push(ti);
            }
            if !adj[ti].contains(&fi) {
                adj[ti].push(fi);
            }
        }
    }

    // Brandes' algorithm for betweenness centrality on undirected graph
    let mut betweenness = vec![0.0f64; n];

    for s in 0..n {
        let mut stack = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n]; // number of shortest paths
        sigma[s] = 1.0;
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                // w found for the first time?
                if dist[w] < 0 {
                    queue.push_back(w);
                    dist[w] = dist[v] + 1;
                }
                // shortest path to w via v?
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    // Normalize (undirected: divide by 2)
    for b in betweenness.iter_mut() {
        *b /= 2.0;
    }

    // Collect intermediate nodes (exclude source and target), sort by betweenness
    let mut candidates: Vec<(usize, f64)> = betweenness
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let path = &node_paths[*i];
            path != source && path != target
        })
        .map(|(i, &b)| (i, b))
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_n);

    candidates
        .into_iter()
        .filter(|(_, b)| *b > 0.0)
        .map(|(i, _)| node_paths[i].clone())
        .collect()
}

/// Compute the density of the bridge subgraph.
///
/// Density = directed_edges / (nodes * (nodes - 1))
/// For undirected interpretation: density = edges / (nodes * (nodes - 1) / 2)
///
/// Returns 0.0 if nodes <= 1.
pub fn compute_bridge_density(node_count: usize, edge_count: usize) -> f64 {
    if node_count <= 1 {
        return 0.0;
    }
    let max_edges = node_count * (node_count - 1);
    edge_count as f64 / max_edges as f64
}

// ============================================================================
// GraIL algorithms — Stress Testing (Plan 5)
// ============================================================================

/// Build an undirected petgraph from our directed CodeGraph.
///
/// Creates a `UnGraph<(), ()>` where node indices match the original graph.
/// An optional `exclude_edge` predicate can filter specific edges.
fn build_undirected<F>(
    g: &petgraph::Graph<super::models::CodeNode, super::models::CodeEdge, petgraph::Directed>,
    exclude_edge: F,
) -> petgraph::graph::UnGraph<(), ()>
where
    F: Fn(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex) -> bool,
{
    let mut ug = petgraph::graph::UnGraph::<(), ()>::with_capacity(g.node_count(), g.edge_count());
    // Add nodes in index order so indices match
    for _ in g.node_indices() {
        ug.add_node(());
    }
    // Add edges (skip excluded)
    for e in g.edge_indices() {
        if let Some((s, t)) = g.edge_endpoints(e) {
            if !exclude_edge(s, t) {
                let s_new = petgraph::graph::NodeIndex::new(s.index());
                let t_new = petgraph::graph::NodeIndex::new(t.index());
                ug.add_edge(s_new, t_new, ());
            }
        }
    }
    ug
}

/// Simulate removing a node from the graph and measure impact.
///
/// Computes WCC before and after removal. Orphaned nodes are those
/// that were in the same component as the target but end up in a
/// singleton component after removal.
///
/// Returns `StressTestResult` with resilience_score = 1.0 - (orphans / total).
pub fn stress_test_node_removal(
    graph: &CodeGraph,
    target_id: &str,
) -> Option<super::models::StressTestResult> {
    use petgraph::algo::connected_components;

    let g = &graph.graph;
    let target_idx = graph.id_to_index.get(target_id)?;

    let total_nodes = g.node_count();
    if total_nodes <= 1 {
        return Some(super::models::StressTestResult {
            target: target_id.to_string(),
            mode: super::models::StressTestMode::NodeRemoval,
            resilience_score: 0.0,
            orphaned_nodes: 0,
            blast_radius: 0,
            cascade_depth: 0,
            components_before: 1,
            components_after: 0,
            critical_edges: vec![],
        });
    }

    // Convert to undirected for WCC
    let undirected = build_undirected(g, |_, _| false);
    let components_before = connected_components(&undirected);

    // BFS on undirected graph, skipping target node, to compute:
    // - components_after (number of WCC excluding target)
    // - orphaned_nodes (singleton components created by removal)
    let target_ug = petgraph::graph::NodeIndex::<u32>::new(target_idx.index());
    let mut component_sizes: Vec<usize> = Vec::new();
    let mut visited = vec![false; undirected.node_count()];
    visited[target_ug.index()] = true; // mark target as visited so we skip it

    for start in undirected.node_indices() {
        if visited[start.index()] {
            continue;
        }
        // BFS to find component size
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start.index()] = true;
        let mut size = 0usize;
        while let Some(current) = queue.pop_front() {
            size += 1;
            for neighbor in undirected.neighbors(current) {
                if !visited[neighbor.index()] {
                    visited[neighbor.index()] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component_sizes.push(size);
    }

    let components_after = component_sizes.len();
    let orphaned_nodes = component_sizes.iter().filter(|&&s| s == 1).count();
    let blast_radius = components_after.saturating_sub(components_before);

    let resilience_score = if total_nodes > 1 {
        1.0 - (orphaned_nodes as f64 / (total_nodes - 1) as f64)
    } else {
        0.0
    };

    Some(super::models::StressTestResult {
        target: target_id.to_string(),
        mode: super::models::StressTestMode::NodeRemoval,
        resilience_score: resilience_score.max(0.0),
        orphaned_nodes,
        blast_radius,
        cascade_depth: 0,
        components_before,
        components_after,
        critical_edges: vec![],
    })
}

/// Find bridge edges using Tarjan's algorithm.
///
/// A bridge is an edge whose removal increases the number of connected components.
/// Uses DFS with discovery time and low-link values.
///
/// Returns Vec<(source_id, target_id)> for all bridges in the graph.
pub fn find_bridges(graph: &CodeGraph) -> Vec<(String, String)> {
    let g = &graph.graph;
    let n = g.node_count();
    if n == 0 {
        return vec![];
    }

    let mut disc = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut timer: u32 = 1;
    let mut bridges: Vec<(NodeIndex, NodeIndex)> = Vec::new();

    // Build undirected adjacency (merge both directions)
    let mut adj: Vec<Vec<NodeIndex>> = vec![Vec::new(); n];
    for edge_idx in g.edge_indices() {
        if let Some((s, t)) = g.edge_endpoints(edge_idx) {
            adj[s.index()].push(t);
            adj[t.index()].push(s);
        }
    }
    // Deduplicate adjacency lists
    for neighbors in &mut adj {
        neighbors.sort_by_key(|n| n.index());
        neighbors.dedup();
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs_bridge(
        u: NodeIndex,
        parent: Option<NodeIndex>,
        adj: &[Vec<NodeIndex>],
        disc: &mut [u32],
        low: &mut [u32],
        visited: &mut [bool],
        timer: &mut u32,
        bridges: &mut Vec<(NodeIndex, NodeIndex)>,
    ) {
        visited[u.index()] = true;
        disc[u.index()] = *timer;
        low[u.index()] = *timer;
        *timer += 1;

        for &v in &adj[u.index()] {
            if Some(v) == parent {
                continue;
            }
            if visited[v.index()] {
                low[u.index()] = low[u.index()].min(disc[v.index()]);
            } else {
                dfs_bridge(v, Some(u), adj, disc, low, visited, timer, bridges);
                low[u.index()] = low[u.index()].min(low[v.index()]);
                if low[v.index()] > disc[u.index()] {
                    bridges.push((u, v));
                }
            }
        }
    }

    // Run DFS from each unvisited node (handles disconnected graphs)
    for node in g.node_indices() {
        if !visited[node.index()] {
            dfs_bridge(
                node,
                None,
                &adj,
                &mut disc,
                &mut low,
                &mut visited,
                &mut timer,
                &mut bridges,
            );
        }
    }

    // Convert to ID strings
    bridges
        .into_iter()
        .map(|(u, v)| (g[u].id.clone(), g[v].id.clone()))
        .collect()
}

/// Simulate cascade removal: iteratively remove orphaned dependents.
///
/// Starting from the target node, removes it, then finds all nodes
/// whose incoming dependencies are ALL in the removed set, removes them too,
/// repeating until no new orphans are found or max_iterations is reached.
///
/// Returns blast_radius (total removed) and cascade_depth.
pub fn stress_test_cascade(
    graph: &CodeGraph,
    target_id: &str,
    max_iterations: usize,
) -> Option<super::models::StressTestResult> {
    let g = &graph.graph;
    let target_idx = *graph.id_to_index.get(target_id)?;
    let total_nodes = g.node_count();

    let mut removed: HashSet<NodeIndex> = HashSet::new();
    removed.insert(target_idx);

    let mut cascade_depth = 0;

    for _ in 0..max_iterations {
        let mut new_orphans: Vec<NodeIndex> = Vec::new();

        for node in g.node_indices() {
            if removed.contains(&node) {
                continue;
            }
            // Check if ALL incoming edges come from removed nodes
            let incoming: Vec<NodeIndex> =
                g.neighbors_directed(node, Direction::Incoming).collect();

            if !incoming.is_empty() && incoming.iter().all(|n| removed.contains(n)) {
                new_orphans.push(node);
            }
        }

        if new_orphans.is_empty() {
            break;
        }

        for orphan in &new_orphans {
            removed.insert(*orphan);
        }
        cascade_depth += 1;
    }

    let blast_radius = removed.len();
    let orphaned_nodes = blast_radius.saturating_sub(1); // exclude the target itself

    let resilience_score = if total_nodes > 1 {
        1.0 - (orphaned_nodes as f64 / (total_nodes - 1) as f64)
    } else {
        0.0
    };

    // Compute WCC before
    let undirected = build_undirected(g, |_, _| false);
    let components_before = petgraph::algo::connected_components(&undirected);

    Some(super::models::StressTestResult {
        target: target_id.to_string(),
        mode: super::models::StressTestMode::Cascade,
        resilience_score: resilience_score.max(0.0),
        orphaned_nodes,
        blast_radius,
        cascade_depth,
        components_before,
        components_after: 0, // Not trivially computed for cascade
        critical_edges: vec![],
    })
}

/// Simulate removing an edge from the graph and measure impact.
///
/// Checks if the edge is a bridge (increases WCC count by 1).
pub fn stress_test_edge_removal(
    graph: &CodeGraph,
    from_id: &str,
    to_id: &str,
) -> Option<super::models::StressTestResult> {
    let g = &graph.graph;
    let _from_idx = graph.id_to_index.get(from_id)?;
    let _to_idx = graph.id_to_index.get(to_id)?;

    // Build undirected and count components before
    let undirected_before = build_undirected(g, |_, _| false);
    let components_before = petgraph::algo::connected_components(&undirected_before);

    // Build undirected WITHOUT the target edge
    let from_idx = *_from_idx;
    let to_idx = *_to_idx;
    let undirected_after = build_undirected(g, |s, t| {
        (s == from_idx && t == to_idx) || (s == to_idx && t == from_idx)
    });
    let components_after = petgraph::algo::connected_components(&undirected_after);

    let is_bridge = components_after > components_before;
    let resilience_score = if is_bridge { 0.0 } else { 1.0 };

    Some(super::models::StressTestResult {
        target: format!("{} -> {}", from_id, to_id),
        mode: super::models::StressTestMode::EdgeRemoval,
        resilience_score,
        orphaned_nodes: 0,
        blast_radius: if is_bridge { 1 } else { 0 },
        cascade_depth: 0,
        components_before,
        components_after,
        critical_edges: if is_bridge {
            vec![(from_id.to_string(), to_id.to_string())]
        } else {
            vec![]
        },
    })
}

// ============================================================================
// GraIL algorithms — Context Cards (Plan 8)
// ============================================================================

/// Compute context cards for all File nodes in the graph.
///
/// Aggregates analytics metrics (PageRank, betweenness, clustering, community),
/// structural DNA, WL hash, and edge counts (imports/calls in/out) into a
/// self-contained `ContextCard` for each file.
///
/// Co-change data is extracted from CO_CHANGED edges. The top-5 co-changers
/// are included in each card.
pub fn compute_context_cards(
    graph: &CodeGraph,
    analytics: &GraphAnalytics,
    dna_map: &HashMap<String, Vec<f64>>,
    wl_hashes: &HashMap<String, u64>,
    fp_map: &HashMap<String, Vec<f64>>,
) -> Vec<super::models::ContextCard> {
    use super::models::{CodeEdgeType, CodeNodeType, ContextCard};

    let g = &graph.graph;
    let now = chrono::Utc::now().to_rfc3339();

    // Build community_id → label lookup
    let community_labels: HashMap<u32, &str> = analytics
        .communities
        .iter()
        .map(|c| (c.id, c.label.as_str()))
        .collect();

    // Extract co-change data for top-5 co-changers
    let co_change_data = extract_co_change_data(graph);

    let mut cards = Vec::new();

    for (node_id, node_idx) in &graph.id_to_index {
        let node = &g[*node_idx];
        if node.node_type != CodeNodeType::File {
            continue;
        }

        // Get analytics metrics
        let metrics = analytics.metrics.get(node_id);
        let pagerank = metrics.map(|m| m.pagerank).unwrap_or(0.0);
        let betweenness = metrics.map(|m| m.betweenness).unwrap_or(0.0);
        let clustering = metrics.map(|m| m.clustering_coefficient).unwrap_or(0.0);
        let community_id = metrics.map(|m| m.community_id).unwrap_or(0);
        let community_label = community_labels
            .get(&community_id)
            .unwrap_or(&"unknown")
            .to_string();

        // Count imports in/out and calls in/out
        let mut imports_out = 0usize;
        let mut imports_in = 0usize;
        let mut calls_out = 0usize;
        let mut calls_in = 0usize;

        // Outgoing edges
        for edge_ref in g.edges(*node_idx) {
            match edge_ref.weight().edge_type {
                CodeEdgeType::Imports => imports_out += 1,
                CodeEdgeType::Calls => calls_out += 1,
                _ => {}
            }
        }

        // Incoming edges
        for edge_ref in g.edges_directed(*node_idx, petgraph::Direction::Incoming) {
            match edge_ref.weight().edge_type {
                CodeEdgeType::Imports => imports_in += 1,
                CodeEdgeType::Calls => calls_in += 1,
                _ => {}
            }
        }

        // Top-5 co-changers
        let mut co_changers: Vec<(String, f64)> = co_change_data
            .iter()
            .filter_map(|((a, b), &weight)| {
                if a == node_id {
                    Some((b.clone(), weight))
                } else if b == node_id {
                    Some((a.clone(), weight))
                } else {
                    None
                }
            })
            .collect();
        co_changers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        co_changers.truncate(5);
        let cc_co_changers_top5: Vec<String> = co_changers.into_iter().map(|(p, _)| p).collect();

        // DNA & WL hash
        let dna = dna_map.get(node_id).cloned().unwrap_or_default();
        let wl = wl_hashes.get(node_id).copied().unwrap_or(0);

        cards.push(ContextCard {
            path: node_id.clone(),
            cc_pagerank: pagerank,
            cc_betweenness: betweenness,
            cc_clustering: clustering,
            cc_community_id: community_id,
            cc_community_label: community_label,
            cc_imports_out: imports_out,
            cc_imports_in: imports_in,
            cc_calls_out: calls_out,
            cc_calls_in: calls_in,
            cc_structural_dna: dna,
            cc_wl_hash: wl,
            cc_fingerprint: fp_map.get(node_id).cloned().unwrap_or_default(),
            cc_co_changers_top5,
            cc_version: 1,
            cc_computed_at: now.clone(),
        });
    }

    cards
}

// ============================================================================
// GraIL algorithms — Missing Link Prediction (Plan 9)
// ============================================================================

/// Find all pairs of nodes at distance exactly 2 (not directly connected).
///
/// Algorithm: for each node, BFS 1-hop → for each neighbor, BFS 1-hop →
/// collect pairs (node, hop2) where hop2 is NOT directly connected to node.
/// Deduplicate by storing (min_index, max_index).
///
/// Complexity: O(V × avg_degree²) — for 10K nodes with avg_degree=5 → ~250K pairs.
pub fn find_distance_2_3_pairs(graph: &CodeGraph) -> Vec<(NodeIndex, NodeIndex)> {
    let g = &graph.graph;
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();

    for node in g.node_indices() {
        let neighbors: HashSet<NodeIndex> = undirected_neighbors(g, node).into_iter().collect();

        // For each neighbor, look at THEIR neighbors (distance 2 from node)
        for &neighbor in &neighbors {
            for hop2 in undirected_neighbors(g, neighbor) {
                // Skip if hop2 is the source node itself
                if hop2 == node {
                    continue;
                }
                // Skip if hop2 is directly connected to node (distance 1, not 2)
                if neighbors.contains(&hop2) {
                    continue;
                }
                // Deduplicate: store as (min, max) index pair
                let pair = if node.index() < hop2.index() {
                    (node.index(), hop2.index())
                } else {
                    (hop2.index(), node.index())
                };
                pairs.insert(pair);
            }
        }
    }

    pairs
        .into_iter()
        .map(|(a, b)| (NodeIndex::new(a), NodeIndex::new(b)))
        .collect()
}

/// Compute plausibility score for a potential link between two nodes.
///
/// Uses 5 signals with weighted fusion:
/// - **Jaccard** (0.25): neighbor set overlap — high = similar connectivity
/// - **Co-change** (0.30): temporal coupling from commit history — strongest signal
/// - **Proximity** (0.15): inverse shortest-path distance — closer = more likely
/// - **Adamic-Adar** (0.15): weighted common neighbors (1/ln(degree)) — penalizes hubs
/// - **DNA similarity** (0.15): structural role similarity via cosine on DNA vectors
///
/// Returns a `LinkPrediction` with the combined score and individual signal values.
pub fn link_plausibility(
    graph: &CodeGraph,
    source: NodeIndex,
    target: NodeIndex,
    co_change_data: &HashMap<(String, String), f64>,
    dna_map: Option<&HashMap<String, Vec<f64>>>,
) -> super::models::LinkPrediction {
    let g = &graph.graph;

    // --- Signal 1: Jaccard coefficient (common neighbors / union neighbors) ---
    let source_neighbors: HashSet<NodeIndex> =
        undirected_neighbors(g, source).into_iter().collect();
    let target_neighbors: HashSet<NodeIndex> =
        undirected_neighbors(g, target).into_iter().collect();
    let common_count = source_neighbors.intersection(&target_neighbors).count();
    let union_count = source_neighbors.union(&target_neighbors).count();
    let jaccard = if union_count > 0 {
        common_count as f64 / union_count as f64
    } else {
        0.0
    };

    // --- Signal 2: Co-change weight (temporal coupling from commits) ---
    let source_id = &g[source].id;
    let target_id = &g[target].id;
    let co_change_weight = co_change_data
        .get(&(source_id.clone(), target_id.clone()))
        .or_else(|| co_change_data.get(&(target_id.clone(), source_id.clone())))
        .copied()
        .unwrap_or(0.0)
        .min(1.0); // Clamp to [0, 1]

    // --- Signal 3: Proximity (inverse shortest-path distance via BFS) ---
    let proximity = {
        // BFS from source to target (unweighted, undirected)
        let mut visited = vec![false; g.node_count()];
        let mut queue = std::collections::VecDeque::new();
        visited[source.index()] = true;
        queue.push_back((source, 0u32));
        let mut distance = u32::MAX;

        while let Some((current, dist)) = queue.pop_front() {
            if current == target {
                distance = dist;
                break;
            }
            if dist >= 10 {
                break; // Cap search depth
            }
            for neighbor in undirected_neighbors(g, current) {
                if !visited[neighbor.index()] {
                    visited[neighbor.index()] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        if distance > 0 && distance < u32::MAX {
            1.0 / distance as f64
        } else {
            0.0
        }
    };

    // --- Signal 4: Adamic-Adar index (penalizes high-degree common neighbors) ---
    let adamic_adar: f64 = source_neighbors
        .intersection(&target_neighbors)
        .map(|&common_node| {
            let degree = undirected_neighbors(g, common_node).len();
            if degree > 1 {
                1.0 / (degree as f64).ln()
            } else {
                0.0
            }
        })
        .sum();
    // Normalize Adamic-Adar to [0, 1] range (cap at reasonable max)
    let adamic_adar_norm = (adamic_adar / 5.0).min(1.0);

    // --- Signal 5: Structural DNA cosine similarity ---
    let dna_similarity = dna_map
        .and_then(|dm| {
            let s_dna = dm.get(source_id)?;
            let t_dna = dm.get(target_id)?;
            Some(cosine_similarity(s_dna, t_dna))
        })
        .unwrap_or(0.0);

    // --- Weighted fusion ---
    let plausibility = 0.25 * jaccard
        + 0.30 * co_change_weight
        + 0.15 * proximity
        + 0.15 * adamic_adar_norm
        + 0.15 * dna_similarity;

    let suggested_relation = infer_relation_type(graph, source, target);

    super::models::LinkPrediction {
        source: source_id.clone(),
        target: target_id.clone(),
        plausibility,
        signals: vec![
            ("jaccard".to_string(), jaccard),
            ("co_change".to_string(), co_change_weight),
            ("proximity".to_string(), proximity),
            ("adamic_adar".to_string(), adamic_adar_norm),
            ("dna_similarity".to_string(), dna_similarity),
        ],
        suggested_relation,
    }
}

/// Suggest the top-N most plausible missing links in the graph.
///
/// Algorithm:
/// 1. Find all pairs at distance 2 (not directly connected)
/// 2. Optionally pre-filter by co-change weight > 0 for efficiency
/// 3. Score each pair with `link_plausibility` (5 signals)
/// 4. Filter by `min_plausibility`, sort descending, truncate to `top_n`
///
/// # Performance
/// O(V × avg_degree²) for candidate finding + O(candidates × avg_degree) for scoring.
/// For 10K nodes with avg_degree=5: ~250K candidates → ~1s total.
pub fn suggest_missing_links(
    graph: &CodeGraph,
    co_change_data: &HashMap<(String, String), f64>,
    dna_map: Option<&HashMap<String, Vec<f64>>>,
    top_n: usize,
    min_plausibility: f64,
) -> Vec<super::models::LinkPrediction> {
    let candidates = find_distance_2_3_pairs(graph);

    let mut predictions: Vec<super::models::LinkPrediction> = candidates
        .into_iter()
        .map(|(s, t)| link_plausibility(graph, s, t, co_change_data, dna_map))
        .filter(|p| p.plausibility >= min_plausibility)
        .collect();

    predictions.sort_by(|a, b| {
        b.plausibility
            .partial_cmp(&a.plausibility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    predictions.truncate(top_n);
    predictions
}

/// Extract co-change weights from CoChanged edges in the graph.
///
/// Returns a HashMap of (source_id, target_id) → weight for all CO_CHANGED edges.
/// Both directions are stored for O(1) bidirectional lookup.
pub fn extract_co_change_data(graph: &CodeGraph) -> HashMap<(String, String), f64> {
    use super::models::CodeEdgeType;

    let g = &graph.graph;
    let mut data = HashMap::new();

    for edge_idx in g.edge_indices() {
        if let Some(edge) = g.edge_weight(edge_idx) {
            if edge.edge_type == CodeEdgeType::CoChanged {
                if let Some((source_idx, target_idx)) = g.edge_endpoints(edge_idx) {
                    let source_id = g[source_idx].id.clone();
                    let target_id = g[target_idx].id.clone();
                    data.insert((source_id.clone(), target_id.clone()), edge.weight);
                    data.insert((target_id, source_id), edge.weight);
                }
            }
        }
    }

    data
}

/// Infer the most likely relation type for a predicted link.
///
/// Uses node types: File→File = IMPORTS, Function→Function = CALLS,
/// mixed or other = RELATED.
pub fn infer_relation_type(graph: &CodeGraph, source: NodeIndex, target: NodeIndex) -> String {
    use super::models::CodeNodeType;

    let source_type = &graph.graph[source].node_type;
    let target_type = &graph.graph[target].node_type;

    match (source_type, target_type) {
        (CodeNodeType::File, CodeNodeType::File) => "IMPORTS".to_string(),
        (CodeNodeType::Function, CodeNodeType::Function) => "CALLS".to_string(),
        (CodeNodeType::File, CodeNodeType::Function)
        | (CodeNodeType::Function, CodeNodeType::File) => "DEFINES".to_string(),
        (CodeNodeType::Struct, CodeNodeType::Trait)
        | (CodeNodeType::Trait, CodeNodeType::Struct) => "IMPLEMENTS_TRAIT".to_string(),
        _ => "RELATED".to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{CodeEdge, CodeEdgeType, CodeNode, CodeNodeType, LargeGraphConfig};

    /// Build a star graph: center → [leaf1, leaf2, ..., leafN]
    fn make_star_graph(n_leaves: usize) -> CodeGraph {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "center".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "center".to_string(),
            project_id: None,
        });
        for i in 0..n_leaves {
            let id = format!("leaf_{}", i);
            g.add_node(CodeNode {
                id: id.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: id.clone(),
                project_id: None,
            });
            g.add_edge(
                "center",
                &id,
                CodeEdge {
                    edge_type: CodeEdgeType::Calls,
                    weight: 1.0,
                },
            );
        }
        g
    }

    /// Build a linear chain: A → B → C → D → E
    fn make_chain_graph(n: usize) -> CodeGraph {
        let mut g = CodeGraph::new();
        let names: Vec<String> = (0..n).map(|i| format!("node_{}", i)).collect();
        for name in &names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }
        for i in 0..n - 1 {
            g.add_edge(
                &names[i],
                &names[i + 1],
                CodeEdge {
                    edge_type: CodeEdgeType::Calls,
                    weight: 1.0,
                },
            );
        }
        g
    }

    /// Build two cliques connected by a single edge (for Louvain testing).
    fn make_two_cliques(size: usize) -> CodeGraph {
        let mut g = CodeGraph::new();

        // Clique A
        let a_names: Vec<String> = (0..size).map(|i| format!("a_{}", i)).collect();
        for name in &a_names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    g.add_edge(
                        &a_names[i],
                        &a_names[j],
                        CodeEdge {
                            edge_type: CodeEdgeType::Calls,
                            weight: 1.0,
                        },
                    );
                }
            }
        }

        // Clique B
        let b_names: Vec<String> = (0..size).map(|i| format!("b_{}", i)).collect();
        for name in &b_names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    g.add_edge(
                        &b_names[i],
                        &b_names[j],
                        CodeEdge {
                            edge_type: CodeEdgeType::Calls,
                            weight: 1.0,
                        },
                    );
                }
            }
        }

        // Single bridge edge between cliques
        g.add_edge(
            &a_names[0],
            &b_names[0],
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );

        g
    }

    /// Build a complete graph K_n
    fn make_complete_graph(n: usize) -> CodeGraph {
        let mut g = CodeGraph::new();
        let names: Vec<String> = (0..n).map(|i| format!("node_{}", i)).collect();
        for name in &names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    g.add_edge(
                        &names[i],
                        &names[j],
                        CodeEdge {
                            edge_type: CodeEdgeType::Calls,
                            weight: 1.0,
                        },
                    );
                }
            }
        }
        g
    }

    /// Build a triangle: A → B → C → A
    fn make_triangle() -> CodeGraph {
        let mut g = CodeGraph::new();
        for name in &["A", "B", "C"] {
            g.add_node(CodeNode {
                id: name.to_string(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "A",
            "B",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g.add_edge(
            "B",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g.add_edge(
            "C",
            "A",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g
    }

    /// Build a graph with two disconnected subgraphs
    fn make_disconnected_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        // Component 1: 3 nodes
        for name in &["c1_a", "c1_b", "c1_c"] {
            g.add_node(CodeNode {
                id: name.to_string(),
                node_type: CodeNodeType::File,
                path: Some(format!("src/{}.rs", name)),
                name: name.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "c1_a",
            "c1_b",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "c1_b",
            "c1_c",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );

        // Component 2: 2 nodes
        for name in &["c2_x", "c2_y"] {
            g.add_node(CodeNode {
                id: name.to_string(),
                node_type: CodeNodeType::File,
                path: Some(format!("lib/{}.rs", name)),
                name: name.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "c2_x",
            "c2_y",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );

        g
    }

    // --- PageRank Tests ---

    #[test]
    fn test_pagerank_reverse_star_center_highest() {
        // Reverse star: all leaves → center (center is the sink)
        // In this topology, center should have the highest PageRank
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "center".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "center".to_string(),
            project_id: None,
        });
        for i in 0..5 {
            let id = format!("leaf_{}", i);
            g.add_node(CodeNode {
                id: id.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: id.clone(),
                project_id: None,
            });
            g.add_edge(
                &id,
                "center",
                CodeEdge {
                    edge_type: CodeEdgeType::Calls,
                    weight: 1.0,
                },
            );
        }

        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);

        assert_eq!(pr.len(), 6); // center + 5 leaves

        let center_score = pr["center"];
        for i in 0..5 {
            let leaf_score = pr[&format!("leaf_{}", i)];
            assert!(
                center_score >= leaf_score,
                "Center ({}) should have highest PageRank, but leaf_{} has {}",
                center_score,
                i,
                leaf_score
            );
        }

        // Sum should be ≈ 1.0
        let total: f64 = pr.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "PageRank sum should be ≈ 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_pagerank_directed_star_leaves_higher() {
        // In a directed star center→leaves, leaves are dangling nodes
        // and actually accumulate more PR than the center
        let g = make_star_graph(5);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);

        assert_eq!(pr.len(), 6);

        // In this directed topology, leaves (dangling) have higher PR
        let center_score = pr["center"];
        let leaf_score = pr["leaf_0"];
        assert!(
            leaf_score > center_score,
            "In directed star center→leaves, leaves should have higher PR: leaf={}, center={}",
            leaf_score,
            center_score
        );

        // Sum should be ≈ 1.0
        let total: f64 = pr.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "PageRank sum should be ≈ 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let g = CodeGraph::new();
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);
        assert!(pr.is_empty());
    }

    // --- Betweenness Tests ---

    #[test]
    fn test_betweenness_chain_middle_highest() {
        let g = make_chain_graph(5);
        let bc = betweenness_centrality(&g);

        assert_eq!(bc.len(), 5);

        // Middle node (node_2) should have highest betweenness in a chain
        let mid_score = bc["node_2"];
        let end_score = bc["node_0"];
        assert!(
            mid_score > end_score,
            "Middle node ({}) should have higher betweenness than end node ({})",
            mid_score,
            end_score
        );
    }

    #[test]
    fn test_betweenness_star_center_highest() {
        let g = make_star_graph(5);
        let bc = betweenness_centrality(&g);

        let center_score = bc["center"];
        for i in 0..5 {
            assert!(center_score >= bc[&format!("leaf_{}", i)]);
        }
    }

    // --- Louvain Tests ---

    #[test]
    fn test_louvain_two_cliques_detects_2_communities() {
        let g = make_two_cliques(4); // Two K4 connected by one edge
        let (node_map, communities, modularity) =
            louvain_communities(&g, &AnalyticsConfig::default());

        assert_eq!(node_map.len(), 8);
        assert!(
            communities.len() == 2,
            "Expected 2 communities, got {}",
            communities.len()
        );
        assert!(modularity > 0.0, "Modularity should be positive");

        // All a_* nodes should be in the same community
        let a_comm = node_map["a_0"];
        for i in 1..4 {
            assert_eq!(
                node_map[&format!("a_{}", i)],
                a_comm,
                "All a_* nodes should be in the same community"
            );
        }

        // All b_* nodes should be in the same community
        let b_comm = node_map["b_0"];
        for i in 1..4 {
            assert_eq!(node_map[&format!("b_{}", i)], b_comm);
        }

        // The two communities should be different
        assert_ne!(
            a_comm, b_comm,
            "The two cliques should be in different communities"
        );
    }

    #[test]
    fn test_louvain_complete_graph_single_community() {
        let g = make_complete_graph(5); // K5
        let (_, communities, _) = louvain_communities(&g, &AnalyticsConfig::default());

        assert_eq!(
            communities.len(),
            1,
            "Complete graph should form 1 community, got {}",
            communities.len()
        );
        assert_eq!(communities[0].size, 5);
    }

    // --- Clustering Coefficient Tests ---

    #[test]
    fn test_clustering_triangle_all_one() {
        let g = make_triangle();
        let cc = clustering_coefficient(&g);

        for (id, coeff) in &cc {
            assert!(
                (*coeff - 1.0).abs() < f64::EPSILON,
                "Node {} in triangle should have clustering coefficient 1.0, got {}",
                id,
                coeff
            );
        }
    }

    #[test]
    fn test_clustering_star_center_zero() {
        let g = make_star_graph(5);
        let cc = clustering_coefficient(&g);

        assert!(
            (cc["center"] - 0.0).abs() < f64::EPSILON,
            "Star center should have clustering coefficient 0.0, got {}",
            cc["center"]
        );
    }

    // --- Connected Components Tests ---

    #[test]
    fn test_connected_components_disconnected() {
        let g = make_disconnected_graph();
        let (node_map, components) = connected_components(&g);

        assert_eq!(node_map.len(), 5);
        assert_eq!(components.len(), 2);

        // Largest component has 3 nodes and is marked as main
        assert_eq!(components[0].size, 3);
        assert!(components[0].is_main);

        // Smaller component has 2 nodes
        assert_eq!(components[1].size, 2);
        assert!(!components[1].is_main);
    }

    #[test]
    fn test_connected_components_single() {
        let g = make_chain_graph(5);
        let (_, components) = connected_components(&g);

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size, 5);
        assert!(components[0].is_main);
    }

    // --- compute_all Tests ---

    #[test]
    fn test_compute_all_assembles_all_metrics() {
        let g = make_two_cliques(4); // 8 nodes, well-structured
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert_eq!(analytics.node_count, 8);
        assert_eq!(analytics.metrics.len(), 8);
        assert!(!analytics.communities.is_empty());
        assert!(!analytics.components.is_empty());
        assert!(analytics.modularity >= 0.0);

        // Every node should have metrics
        for m in analytics.metrics.values() {
            assert!(m.pagerank >= 0.0);
            assert!(m.betweenness >= 0.0);
            assert!(m.clustering_coefficient >= 0.0);
        }
    }

    #[test]
    fn test_compute_all_empty_graph() {
        let g = CodeGraph::new();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert_eq!(analytics.node_count, 0);
        assert_eq!(analytics.edge_count, 0);
        assert!(analytics.metrics.is_empty());
        assert!(analytics.communities.is_empty());
        assert!(analytics.components.is_empty());
    }

    // --- Benchmark Test ---

    #[test]
    fn test_benchmark_compute_all_500_nodes() {
        // Generate a synthetic graph with ~500 nodes and ~2000 edges
        let mut g = CodeGraph::with_capacity(500, 2000);
        let names: Vec<String> = (0..500).map(|i| format!("func_{}", i)).collect();
        for name in &names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }

        // Create ~2000 edges using a deterministic pattern
        let mut edge_count = 0;
        for i in 0..500 {
            // Each node connects to ~4 others (deterministic, not random)
            for offset in &[1, 7, 31, 127] {
                let j = (i + offset) % 500;
                if i != j {
                    g.add_edge(
                        &names[i],
                        &names[j],
                        CodeEdge {
                            edge_type: CodeEdgeType::Calls,
                            weight: 1.0,
                        },
                    );
                    edge_count += 1;
                }
            }
        }

        assert!(
            edge_count >= 1900,
            "Expected ~2000 edges, got {}",
            edge_count
        );

        let config = AnalyticsConfig::default();
        let start = std::time::Instant::now();
        let analytics = compute_all(&g, &config);
        let elapsed = start.elapsed();

        assert_eq!(analytics.node_count, 500);
        assert_eq!(analytics.metrics.len(), 500);
        assert!(!analytics.communities.is_empty());

        // Performance check: should complete in < 100ms in release mode.
        // Debug builds are ~10-20x slower; use 5000ms as the limit.
        assert!(
            elapsed.as_millis() < 5000,
            "compute_all on 500 nodes took {}ms (limit: 5000ms for debug build)",
            elapsed.as_millis()
        );

        eprintln!(
            "Benchmark: {} nodes, {} edges → {}ms",
            analytics.node_count,
            analytics.edge_count,
            elapsed.as_millis()
        );
    }

    // --- Community Label Tests ---

    #[test]
    fn test_community_label_common_path() {
        let members = vec![
            "src/api/handlers.rs".to_string(),
            "src/api/routes.rs".to_string(),
            "src/api/middleware.rs".to_string(),
        ];
        let label = generate_community_label(&members);
        assert_eq!(label, "api");
    }

    #[test]
    fn test_community_label_no_common_path() {
        let members = vec![
            "src/api/handlers.rs".to_string(),
            "lib/utils/helpers.rs".to_string(),
        ];
        let label = generate_community_label(&members);
        // No common prefix → falls back to "group_N"
        assert!(label.starts_with("group_"), "Got: {}", label);
    }

    // --- Health Report Tests ---

    #[test]
    fn test_health_detects_cycle() {
        let g = make_triangle(); // A→B→C→A is a cycle
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert!(
            !analytics.health.circular_dependencies.is_empty(),
            "Triangle should be detected as circular dependency"
        );
    }

    #[test]
    fn test_health_detects_orphan_files() {
        let mut g = CodeGraph::new();
        // Connected pair
        g.add_node(CodeNode {
            id: "src/a.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/a.rs".to_string()),
            name: "a.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "src/b.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/b.rs".to_string()),
            name: "b.rs".to_string(),
            project_id: None,
        });
        g.add_edge(
            "src/a.rs",
            "src/b.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        // Orphan file
        g.add_node(CodeNode {
            id: "src/orphan.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/orphan.rs".to_string()),
            name: "orphan.rs".to_string(),
            project_id: None,
        });

        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert!(
            analytics
                .health
                .orphan_files
                .contains(&"src/orphan.rs".to_string()),
            "src/orphan.rs should be detected as orphan, got: {:?}",
            analytics.health.orphan_files
        );
    }

    // --- Large-Graph Mode Tests ---

    /// Build a graph with mixed edge weights (some low, some high) for
    /// testing the large-graph edge filtering.
    fn make_mixed_weight_graph(n: usize) -> CodeGraph {
        let mut g = CodeGraph::new();
        let names: Vec<String> = (0..n).map(|i| format!("node_{}", i)).collect();
        for name in &names {
            g.add_node(CodeNode {
                id: name.clone(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.clone(),
                project_id: None,
            });
        }
        // Create two dense subgraphs connected by low-weight edges:
        // First half: strong edges (weight 0.9)
        let half = n / 2;
        for i in 0..half {
            for j in (i + 1)..half {
                g.add_edge(
                    &names[i],
                    &names[j],
                    CodeEdge {
                        edge_type: CodeEdgeType::Calls,
                        weight: 0.9,
                    },
                );
            }
        }
        // Second half: strong edges (weight 0.9)
        for i in half..n {
            for j in (i + 1)..n {
                g.add_edge(
                    &names[i],
                    &names[j],
                    CodeEdge {
                        edge_type: CodeEdgeType::Calls,
                        weight: 0.9,
                    },
                );
            }
        }
        // Cross-group: low-weight edges (weight 0.3)
        for i in 0..half {
            let j = half + (i % (n - half));
            g.add_edge(
                &names[i],
                &names[j],
                CodeEdge {
                    edge_type: CodeEdgeType::Calls,
                    weight: 0.3,
                },
            );
        }
        g
    }

    #[test]
    fn test_louvain_large_graph_none_identical_to_classic() {
        // With large_graph = None, results must be identical to classic mode
        let g = make_two_cliques(4);
        let config_none = AnalyticsConfig::default(); // large_graph: None
        let (map1, comms1, mod1) = louvain_communities(&g, &config_none);

        // Explicitly set large_graph but with a threshold above the graph size
        let config_high = AnalyticsConfig {
            large_graph: Some(LargeGraphConfig {
                max_nodes_full: 1000, // 8 nodes < 1000 → not activated
                ..Default::default()
            }),
            ..Default::default()
        };
        let (map2, comms2, mod2) = louvain_communities(&g, &config_high);

        assert_eq!(map1, map2, "Community assignments should be identical");
        assert_eq!(comms1.len(), comms2.len());
        assert!(
            (mod1 - mod2).abs() < 1e-10,
            "Modularity should be identical"
        );
    }

    #[test]
    fn test_louvain_large_graph_filters_low_weight_edges() {
        // Create a graph where low-weight cross-edges link two clusters.
        // With large-graph mode (threshold low enough to activate), those
        // low-weight edges should be filtered, resulting in cleaner separation.
        let g = make_mixed_weight_graph(20); // 10 + 10 nodes

        // Classic mode (no filtering) — may merge clusters
        let config_classic = AnalyticsConfig::default();
        let (_, comms_classic, _) = louvain_communities(&g, &config_classic);

        // Large-graph mode with threshold = 5 (20 > 5 → activated), min_confidence = 0.5
        let config_lg = AnalyticsConfig {
            large_graph: Some(LargeGraphConfig {
                max_nodes_full: 5,
                min_confidence: 0.5,
                skip_degree_one: true,
                max_duration_ms: 60_000,
            }),
            ..Default::default()
        };
        let (map_lg, comms_lg, _) = louvain_communities(&g, &config_lg);

        // With filtering, the two halves should clearly separate into 2 communities
        assert!(
            comms_lg.len() >= 2,
            "Large-graph mode should detect ≥2 communities (got {}), classic had {}",
            comms_lg.len(),
            comms_classic.len()
        );

        // Verify: first half and second half are in different communities
        let comm_first = map_lg["node_0"];
        let comm_second = map_lg["node_10"];
        assert_ne!(
            comm_first, comm_second,
            "The two halves should be in different communities with edge filtering"
        );
    }

    #[test]
    fn test_louvain_large_graph_timeout_returns_partial() {
        // With max_duration_ms = 0, the loop should break immediately
        // and return a valid (partial) result
        let g = make_two_cliques(4);
        let config = AnalyticsConfig {
            large_graph: Some(LargeGraphConfig {
                max_nodes_full: 1, // 8 > 1 → activated
                min_confidence: 0.0,
                skip_degree_one: false,
                max_duration_ms: 0, // instant timeout
            }),
            ..Default::default()
        };
        let (node_map, communities, _modularity) = louvain_communities(&g, &config);

        // Should still return valid data (every node assigned to a community)
        assert_eq!(node_map.len(), 8, "All 8 nodes should be assigned");
        assert!(
            !communities.is_empty(),
            "Should return at least one community"
        );
        // The community count may vary (partial result), but it should be valid
        let total_members: usize = communities.iter().map(|c| c.size).sum();
        assert_eq!(
            total_members, 8,
            "Total members across communities should equal node count"
        );
    }

    #[test]
    fn test_louvain_large_graph_degree_one_pre_assignment() {
        // Build a star graph: center has 5 neighbors, each leaf has degree 1
        // With skip_degree_one, leaves should be pre-assigned to center's community
        let g = make_star_graph(5);
        let config = AnalyticsConfig {
            large_graph: Some(LargeGraphConfig {
                max_nodes_full: 1, // 6 > 1 → activated
                min_confidence: 0.0,
                skip_degree_one: true,
                max_duration_ms: 60_000,
            }),
            ..Default::default()
        };
        let (node_map, communities, _) = louvain_communities(&g, &config);

        // All leaves should be in the same community as center
        let center_comm = node_map["center"];
        for i in 0..5 {
            assert_eq!(
                node_map[&format!("leaf_{}", i)],
                center_comm,
                "Leaf {} should be pre-assigned to center's community",
                i
            );
        }
        // Should be a single community
        assert_eq!(communities.len(), 1, "Star graph should form 1 community");
    }

    // --- Cohesion Tests ---

    #[test]
    fn test_cohesion_two_cliques_high() {
        let g = make_two_cliques(4);
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        // Two well-separated cliques should each have high cohesion
        assert_eq!(analytics.communities.len(), 2);
        for comm in &analytics.communities {
            assert!(
                comm.cohesion > 0.8,
                "Community {} (size {}) should have high cohesion, got {}",
                comm.id,
                comm.size,
                comm.cohesion
            );
        }
    }

    #[test]
    fn test_cohesion_single_node_community() {
        // A single isolated node forms its own community → cohesion = 1.0
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "alone".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "alone".to_string(),
            project_id: None,
        });

        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert_eq!(analytics.communities.len(), 1);
        assert!(
            (analytics.communities[0].cohesion - 1.0).abs() < f64::EPSILON,
            "Single-node community should have cohesion 1.0, got {}",
            analytics.communities[0].cohesion
        );
    }

    #[test]
    fn test_cohesion_complete_graph_is_one() {
        // Complete graph K5 → 1 community → all edges internal → cohesion = 1.0
        let g = make_complete_graph(5);
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        assert_eq!(analytics.communities.len(), 1);
        assert!(
            (analytics.communities[0].cohesion - 1.0).abs() < f64::EPSILON,
            "Complete graph single community should have cohesion 1.0, got {}",
            analytics.communities[0].cohesion
        );
    }

    #[test]
    fn test_cohesion_compute_all_includes_cohesion() {
        // Verify that compute_all populates cohesion on every community
        let g = make_two_cliques(4);
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        for comm in &analytics.communities {
            assert!(
                comm.cohesion > 0.0,
                "Community {} should have non-zero cohesion",
                comm.id
            );
        }
    }

    // --- Analysis Profiles (Plan 6) ---

    /// Helper to build a simple test graph with IMPORTS and CALLS edges.
    fn build_profile_test_graph() -> CodeGraph {
        use super::super::models::{CodeEdge, CodeEdgeType, CodeNode, CodeNodeType};

        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "a.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("a.rs".to_string()),
            name: "a.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "b.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("b.rs".to_string()),
            name: "b.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "c.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("c.rs".to_string()),
            name: "c.rs".to_string(),
            project_id: None,
        });
        // a -> b via IMPORTS (weight 1.0)
        g.add_edge(
            "a.rs",
            "b.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        // b -> c via CALLS (weight 1.0)
        g.add_edge(
            "b.rs",
            "c.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        // c -> a via IMPORTS (weight 1.0) — cycle
        g.add_edge(
            "c.rs",
            "a.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g
    }

    #[test]
    fn test_apply_profile_weights_zeroes_imports() {
        use super::super::models::profile_security;

        let g = build_profile_test_graph();
        // Security profile has IMPORTS=0.4, CALLS=0.9
        let security = profile_security();
        let weighted = apply_profile_weights(&g, &security);

        // Check edge weights were multiplied
        for edge in weighted.graph.edge_references() {
            let e = edge.weight();
            match e.edge_type.to_string().as_str() {
                "IMPORTS" => assert!(
                    (e.weight - 0.4).abs() < f64::EPSILON,
                    "IMPORTS edge should be 1.0 * 0.4 = 0.4, got {}",
                    e.weight
                ),
                "CALLS" => assert!(
                    (e.weight - 0.9).abs() < f64::EPSILON,
                    "CALLS edge should be 1.0 * 0.9 = 0.9, got {}",
                    e.weight
                ),
                _ => {}
            }
        }
    }

    #[test]
    fn test_apply_profile_weights_preserves_node_count() {
        use super::super::models::profile_refactoring;

        let g = build_profile_test_graph();
        let weighted = apply_profile_weights(&g, &profile_refactoring());
        assert_eq!(g.node_count(), weighted.node_count());
        assert_eq!(g.edge_count(), weighted.edge_count());
    }

    #[test]
    fn test_apply_profile_weights_default_for_unknown_edge_type() {
        use super::super::models::{AnalysisProfile, FusionWeights};

        let g = build_profile_test_graph();
        // Profile with NO edge weights — all should use default 0.5
        let empty_profile = AnalysisProfile {
            id: "test".to_string(),
            project_id: None,
            name: "empty".to_string(),
            description: None,
            edge_weights: HashMap::new(),
            fusion_weights: FusionWeights::default(),
            is_builtin: false,
        };
        let weighted = apply_profile_weights(&g, &empty_profile);

        for edge in weighted.graph.edge_references() {
            assert!(
                (edge.weight().weight - 0.5).abs() < f64::EPSILON,
                "Unknown edge type should use default 0.5, got {}",
                edge.weight().weight
            );
        }
    }

    #[test]
    fn test_compute_all_with_profile_sets_name() {
        use super::super::models::profile_security;

        let g = build_profile_test_graph();
        let config = AnalyticsConfig::default();
        let result = compute_all_with_profile(&g, &config, &profile_security());
        assert_eq!(result.profile_name.as_deref(), Some("security"));
    }

    #[test]
    fn test_compute_all_with_profile_weighted_graph_differs() {
        use super::super::models::{profile_default, profile_security};

        // Verify that apply_profile_weights produces different edge weights
        // for different profiles (which is the precondition for weighted
        // algorithms to produce different results).
        let g = build_profile_test_graph();

        let default_weighted = apply_profile_weights(&g, &profile_default());
        let security_weighted = apply_profile_weights(&g, &profile_security());

        // Collect edge weights from both
        let default_weights: Vec<f64> = default_weighted
            .graph
            .edge_references()
            .map(|e| e.weight().weight)
            .collect();
        let security_weights: Vec<f64> = security_weighted
            .graph
            .edge_references()
            .map(|e| e.weight().weight)
            .collect();

        assert_ne!(
            default_weights, security_weights,
            "Different profiles must produce different edge weights"
        );

        // Also verify compute_all_with_profile runs without error
        let config = AnalyticsConfig::default();
        let result = compute_all_with_profile(&g, &config, &profile_security());
        assert_eq!(result.node_count, 3);
        assert_eq!(result.profile_name.as_deref(), Some("security"));
    }

    // --- Margin Ranking (Plan 10) ---

    #[test]
    fn test_cluster_label_gradient() {
        assert_eq!(cluster_label(0), "critical");
        assert_eq!(cluster_label(1), "high");
        assert_eq!(cluster_label(2), "moderate");
        assert_eq!(cluster_label(3), "low");
        assert_eq!(cluster_label(4), "peripheral");
        assert_eq!(cluster_label(99), "peripheral");
    }

    #[test]
    fn test_detect_natural_clusters_three_groups() {
        // Scores: [0.9, 0.85, 0.5, 0.45, 0.1]
        // Range = 0.8, threshold = 0.15 * 0.8 = 0.12
        // Gaps: 0.05 (no), 0.35 (yes!), 0.05 (no), 0.35 (yes!)
        // → 3 clusters: [0.9, 0.85], [0.5, 0.45], [0.1]
        let scores: Vec<(String, f64)> = vec![
            ("a".into(), 0.9),
            ("b".into(), 0.85),
            ("c".into(), 0.5),
            ("d".into(), 0.45),
            ("e".into(), 0.1),
        ];
        let clusters = detect_natural_clusters(&scores, 0.15);
        assert_eq!(
            clusters.len(),
            3,
            "Expected 3 clusters, got {}",
            clusters.len()
        );

        // Cluster 1: ranks 1-2, avg ~0.875
        assert_eq!(clusters[0].start_rank, 1);
        assert_eq!(clusters[0].end_rank, 2);
        assert!((clusters[0].avg_score - 0.875).abs() < 0.001);
        assert_eq!(clusters[0].label, "critical");

        // Cluster 2: ranks 3-4, avg ~0.475
        assert_eq!(clusters[1].start_rank, 3);
        assert_eq!(clusters[1].end_rank, 4);
        assert!((clusters[1].avg_score - 0.475).abs() < 0.001);
        assert_eq!(clusters[1].label, "high");

        // Cluster 3: rank 5, avg 0.1
        assert_eq!(clusters[2].start_rank, 5);
        assert_eq!(clusters[2].end_rank, 5);
        assert!((clusters[2].avg_score - 0.1).abs() < 0.001);
        assert_eq!(clusters[2].label, "moderate");
    }

    #[test]
    fn test_detect_natural_clusters_single_item() {
        let scores: Vec<(String, f64)> = vec![("a".into(), 0.5)];
        let clusters = detect_natural_clusters(&scores, 0.15);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].start_rank, 1);
        assert_eq!(clusters[0].end_rank, 1);
    }

    #[test]
    fn test_detect_natural_clusters_empty() {
        let scores: Vec<(String, f64)> = vec![];
        let clusters = detect_natural_clusters(&scores, 0.15);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_detect_natural_clusters_all_equal() {
        let scores: Vec<(String, f64)> =
            vec![("a".into(), 0.5), ("b".into(), 0.5), ("c".into(), 0.5)];
        let clusters = detect_natural_clusters(&scores, 0.15);
        assert_eq!(clusters.len(), 1, "Equal scores → single cluster");
    }

    #[test]
    fn test_into_ranked_basic() {
        // 5 items with varying scores
        let items: Vec<(String, f64)> = vec![
            ("c".into(), 0.5),
            ("a".into(), 0.9),
            ("e".into(), 0.1),
            ("b".into(), 0.85),
            ("d".into(), 0.45),
        ];
        let ranked = into_ranked(items, 100);

        assert_eq!(ranked.items.len(), 5);
        assert_eq!(ranked.total_candidates, 100);
        assert!((ranked.score_range.0 - 0.1).abs() < 0.001); // min
        assert!((ranked.score_range.1 - 0.9).abs() < 0.001); // max

        // Verify sorted descending
        assert_eq!(ranked.items[0].rank, 1);
        assert_eq!(ranked.items[0].item, "a"); // score 0.9
        assert_eq!(ranked.items[1].rank, 2);
        assert_eq!(ranked.items[1].item, "b"); // score 0.85
        assert_eq!(ranked.items[2].rank, 3);
        assert_eq!(ranked.items[2].item, "c"); // score 0.5
        assert_eq!(ranked.items[3].rank, 4);
        assert_eq!(ranked.items[3].item, "d"); // score 0.45
        assert_eq!(ranked.items[4].rank, 5);
        assert_eq!(ranked.items[4].item, "e"); // score 0.1

        // Verify margins
        let m0 = ranked.items[0].margin_to_next.unwrap();
        assert!((m0 - 0.05).abs() < 0.001, "margin 0.9→0.85 = 0.05");
        assert!(ranked.items[0].margin_to_prev.is_none()); // first item

        let m4 = ranked.items[4].margin_to_prev.unwrap();
        assert!((m4 - 0.35).abs() < 0.001, "margin 0.45→0.1 = 0.35");
        assert!(ranked.items[4].margin_to_next.is_none()); // last item
    }

    #[test]
    fn test_into_ranked_empty() {
        let items: Vec<(String, f64)> = vec![];
        let ranked = into_ranked(items, 0);
        assert!(ranked.items.is_empty());
        assert!(ranked.natural_clusters.is_empty());
    }

    #[test]
    fn test_into_ranked_confidence_levels() {
        // Item with large margin → High confidence
        // Item with small margin → Low confidence
        // Item with zero margin → Tied
        let items: Vec<(String, f64)> = vec![
            ("a".into(), 1.0),
            ("b".into(), 0.5), // margin_to_prev=0.5 (High), margin_to_next=0.5 (High)
            ("c".into(), 0.0),
        ];
        let ranked = into_ranked(items, 3);

        assert_eq!(ranked.items[0].confidence, RankConfidence::High); // margin_to_next=0.5
        assert_eq!(ranked.items[1].confidence, RankConfidence::High); // min(0.5, 0.5)=0.5
        assert_eq!(ranked.items[2].confidence, RankConfidence::High); // margin_to_prev=0.5
    }

    #[test]
    fn test_into_ranked_serde_json() {
        // Note: #[serde(flatten)] requires T to be a struct/map, not a primitive.
        // Use a simple test struct to verify JSON serialization.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        struct TestItem {
            path: String,
        }

        let items: Vec<(TestItem, f64)> = vec![
            (
                TestItem {
                    path: "file_a.rs".into(),
                },
                0.95,
            ),
            (
                TestItem {
                    path: "file_b.rs".into(),
                },
                0.30,
            ),
        ];
        let ranked = into_ranked(items, 50);
        let json = serde_json::to_value(&ranked).unwrap();

        // Verify top-level structure
        assert!(json["items"].is_array());
        assert_eq!(json["total_candidates"], 50);
        assert!(json["score_range"].is_array());

        // Verify first item has ranking fields flattened with item fields
        let first = &json["items"][0];
        assert_eq!(first["rank"], 1);
        assert_eq!(first["score"], 0.95);
        assert!(first["margin_to_next"].is_number());
        assert!(first["confidence"].is_string());
        // Flattened: item fields at top level
        assert_eq!(first["path"], "file_a.rs");
    }

    #[test]
    fn test_into_ranked_performance_1000_items() {
        // Verify that into_ranked with 1000 items completes quickly
        let items: Vec<(String, f64)> = (0..1000)
            .map(|i| (format!("item_{}", i), i as f64 / 1000.0))
            .collect();

        let start = std::time::Instant::now();
        let ranked = into_ranked(items, 10000);
        let elapsed = start.elapsed();

        assert_eq!(ranked.items.len(), 1000);
        assert!(
            elapsed.as_millis() < 50,
            "into_ranked(1000) took {}ms, expected < 50ms",
            elapsed.as_millis()
        );
    }

    // ========================================================================
    // Bridge subgraph tests (GraIL Plan 1)
    // ========================================================================

    #[test]
    fn test_double_radius_label_linear_graph() {
        // A → B → C, source=A, target=C
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];
        let labels = double_radius_label(&nodes, &edges, "A", "C");

        assert_eq!(labels["A"], (0, 2)); // A: 0 from source, 2 from target
        assert_eq!(labels["B"], (1, 1)); // B: 1 from both
        assert_eq!(labels["C"], (2, 0)); // C: 2 from source, 0 from target
    }

    #[test]
    fn test_double_radius_label_diamond_graph() {
        // S → A → T
        // S → B → T
        let nodes = vec![
            "S".to_string(),
            "A".to_string(),
            "B".to_string(),
            "T".to_string(),
        ];
        let edges = vec![
            ("S".to_string(), "A".to_string()),
            ("S".to_string(), "B".to_string()),
            ("A".to_string(), "T".to_string()),
            ("B".to_string(), "T".to_string()),
        ];
        let labels = double_radius_label(&nodes, &edges, "S", "T");

        assert_eq!(labels["S"], (0, 2));
        assert_eq!(labels["A"], (1, 1));
        assert_eq!(labels["B"], (1, 1));
        assert_eq!(labels["T"], (2, 0));
    }

    #[test]
    fn test_double_radius_label_unreachable_node() {
        // A → B, C is isolated
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![("A".to_string(), "B".to_string())];
        let labels = double_radius_label(&nodes, &edges, "A", "B");

        assert_eq!(labels["A"], (0, 1));
        assert_eq!(labels["B"], (1, 0));
        assert_eq!(labels["C"], (u32::MAX, u32::MAX));
    }

    #[test]
    fn test_find_bottleneck_diamond() {
        // S → A → T, S → B → T: A and B are bottlenecks
        let nodes = vec![
            "S".to_string(),
            "A".to_string(),
            "B".to_string(),
            "T".to_string(),
        ];
        let edges = vec![
            ("S".to_string(), "A".to_string()),
            ("S".to_string(), "B".to_string()),
            ("A".to_string(), "T".to_string()),
            ("B".to_string(), "T".to_string()),
        ];
        let bottlenecks = find_bottleneck_nodes(&nodes, &edges, "S", "T", 3);

        // A and B should both be bottlenecks with equal betweenness
        assert_eq!(bottlenecks.len(), 2);
        assert!(bottlenecks.contains(&"A".to_string()));
        assert!(bottlenecks.contains(&"B".to_string()));
    }

    #[test]
    fn test_find_bottleneck_chain() {
        // A → B → C → D → E, source=A, target=E
        // B, C, D are intermediate; C has highest betweenness (center)
        let nodes: Vec<String> = vec!["A", "B", "C", "D", "E"]
            .into_iter()
            .map(String::from)
            .collect();
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
            ("C".to_string(), "D".to_string()),
            ("D".to_string(), "E".to_string()),
        ];
        let bottlenecks = find_bottleneck_nodes(&nodes, &edges, "A", "E", 1);

        // C has highest betweenness in a chain
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0], "C");
    }

    #[test]
    fn test_find_bottleneck_too_few_nodes() {
        let nodes = vec!["A".to_string(), "B".to_string()];
        let edges = vec![("A".to_string(), "B".to_string())];
        let bottlenecks = find_bottleneck_nodes(&nodes, &edges, "A", "B", 3);
        assert!(bottlenecks.is_empty());
    }

    #[test]
    fn test_compute_bridge_density() {
        // 4 nodes, 6 directed edges → density = 6 / (4*3) = 0.5
        assert!((compute_bridge_density(4, 6) - 0.5).abs() < f64::EPSILON);

        // 3 nodes, 6 edges (fully connected directed) → 6 / 6 = 1.0
        assert!((compute_bridge_density(3, 6) - 1.0).abs() < f64::EPSILON);

        // Edge cases
        assert_eq!(compute_bridge_density(0, 0), 0.0);
        assert_eq!(compute_bridge_density(1, 0), 0.0);
        assert_eq!(compute_bridge_density(2, 1), 0.5);
    }

    // ========================================================================
    // Structural DNA tests
    // ========================================================================

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        assert!((cosine_similarity(&[1.0, 1.0], &[1.0, 1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_structural_dna_empty_graph() {
        let g = CodeGraph::new();
        let pr = HashMap::new();
        let dna = structural_dna(&g, &pr, 5).unwrap();
        assert!(dna.is_empty());
    }

    #[test]
    fn test_structural_dna_chain_graph() {
        // Chain: node_0 → node_1 → node_2 → node_3 → node_4
        let g = make_chain_graph(5);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);

        let dna = structural_dna(&g, &pr, 2).unwrap();

        // All 5 nodes should have DNA
        assert_eq!(dna.len(), 5);

        // Each DNA vector should have K=2 dimensions
        for v in dna.values() {
            assert_eq!(v.len(), 2);
        }

        // All values should be in [0, 1]
        for v in dna.values() {
            for &d in v {
                assert!((0.0..=1.0).contains(&d), "DNA value out of range: {}", d);
            }
        }
    }

    #[test]
    fn test_structural_dna_symmetric_nodes_similar() {
        // Complete graph K5: all nodes are structurally equivalent
        let g = make_complete_graph(5);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);

        let dna = structural_dna(&g, &pr, 2).unwrap();
        assert_eq!(dna.len(), 5);

        // In K5, anchor nodes have dist=0 to themselves → DNA like [0,1] or [1,0].
        // Non-anchor nodes have dist=1 to all anchors → DNA like [1,1].
        // So non-anchor nodes should be perfectly similar (cosine = 1.0),
        // while anchor-to-non-anchor similarity is ~0.707 (cos([0,1],[1,1])).
        // Filter to non-anchor nodes only (those with DNA [1.0, 1.0] normalized)
        let non_anchor_vecs: Vec<&Vec<f64>> = dna
            .values()
            .filter(|v| v.iter().all(|d| *d > 0.0)) // exclude anchors (have a 0.0 dim)
            .collect();

        assert!(
            non_anchor_vecs.len() >= 2,
            "Expected at least 2 non-anchor nodes"
        );
        for i in 0..non_anchor_vecs.len() {
            for j in (i + 1)..non_anchor_vecs.len() {
                let sim = cosine_similarity(non_anchor_vecs[i], non_anchor_vecs[j]);
                assert!(
                    sim > 0.99,
                    "Expected near-perfect similarity between non-anchor K5 nodes, got {}",
                    sim
                );
            }
        }
    }

    #[test]
    fn test_structural_dna_two_cliques_distinct() {
        // Two cliques of size 4 connected by a single bridge
        let g = make_two_cliques(4);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);

        let dna = structural_dna(&g, &pr, 3).unwrap();
        assert_eq!(dna.len(), 8); // 4+4 nodes

        // Nodes within the same clique should be more similar than across cliques
        let a0 = &dna["a_0"];
        let a1 = &dna["a_1"];
        let b2 = &dna["b_2"];

        let intra_sim = cosine_similarity(a0, a1);
        let inter_sim = cosine_similarity(a0, b2);

        // Intra-clique similarity should be higher than inter-clique
        assert!(
            intra_sim > inter_sim,
            "Expected intra({}) > inter({})",
            intra_sim,
            inter_sim
        );
    }

    #[test]
    fn test_structural_dna_no_pagerank_scores() {
        let g = make_chain_graph(3);
        let pr = HashMap::new(); // empty PageRank
        let result = structural_dna(&g, &pr, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_structural_twins_basic() {
        // Use a chain graph where middle nodes have similar structural positions
        let g = make_chain_graph(7);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);
        let dna = structural_dna(&g, &pr, 2).unwrap();

        // Ask for twins of node_3 (center of chain)
        let twins = find_structural_twins(&dna, "node_3", 3);
        assert_eq!(twins.len(), 3);

        // All results should have valid similarity scores in [0, 1]
        for (_, sim) in &twins {
            assert!(
                *sim >= 0.0 && *sim <= 1.0,
                "Similarity out of range: {}",
                sim
            );
        }

        // Results should be sorted descending by similarity
        for i in 0..twins.len() - 1 {
            assert!(twins[i].1 >= twins[i + 1].1);
        }
    }

    #[test]
    fn test_find_structural_twins_nonexistent_target() {
        let dna: HashMap<String, Vec<f64>> = HashMap::new();
        let twins = find_structural_twins(&dna, "nonexistent", 5);
        assert!(twins.is_empty());
    }

    #[test]
    fn test_find_structural_twins_top_n_truncation() {
        let g = make_star_graph(10);
        let config = AnalyticsConfig::default();
        let pr = pagerank(&g, &config);
        let dna = structural_dna(&g, &pr, 2).unwrap();

        let twins = find_structural_twins(&dna, "leaf_0", 3);
        assert_eq!(twins.len(), 3);

        // Results should be sorted by similarity desc
        for i in 0..twins.len() - 1 {
            assert!(twins[i].1 >= twins[i + 1].1);
        }
    }

    // ========================================================================
    // Multi-signal similarity tests
    // ========================================================================

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler_similarity("handlers", "handlers") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_winkler_similar_names() {
        let sim = jaro_winkler_similarity("handlers", "handler");
        assert!(sim > 0.9, "Similar names should have high similarity: {sim}");
    }

    #[test]
    fn test_jaro_winkler_different_names() {
        let sim = jaro_winkler_similarity("handlers", "models");
        assert!(sim < 0.7, "Different names should have low similarity: {sim}");
    }

    #[test]
    fn test_jaro_winkler_empty() {
        assert_eq!(jaro_winkler_similarity("", "test"), 0.0);
        assert_eq!(jaro_winkler_similarity("test", ""), 0.0);
        assert_eq!(jaro_winkler_similarity("", ""), 1.0);
    }

    #[test]
    fn test_jaro_winkler_prefix_boost() {
        // "user_profile" vs "user_settings" share "user_" prefix → boosted
        let with_prefix = jaro_winkler_similarity("user_profile", "user_settings");
        // "profile_user" vs "settings_user" share no prefix → not boosted
        let without_prefix = jaro_winkler_similarity("profile_user", "settings_user");
        assert!(
            with_prefix > without_prefix,
            "Common prefix should boost: {with_prefix} vs {without_prefix}"
        );
    }

    #[test]
    fn test_file_stem() {
        assert_eq!(file_stem("src/api/handlers.rs"), "handlers");
        assert_eq!(file_stem("components/UserProfile.tsx"), "UserProfile");
        assert_eq!(file_stem("Makefile"), "Makefile");
        assert_eq!(file_stem("src/mod.rs"), "mod");
        assert_eq!(file_stem("a/b/c/deep.file.ext"), "deep");
    }

    #[test]
    fn test_log_size_similarity_identical() {
        let sim = log_size_similarity(10, 10);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Same size should be 1.0: {sim}"
        );
    }

    #[test]
    fn test_log_size_similarity_close() {
        let sim = log_size_similarity(10, 12);
        assert!(
            sim > 0.9,
            "Close sizes should have high similarity: {sim}"
        );
    }

    #[test]
    fn test_log_size_similarity_different() {
        let sim = log_size_similarity(1, 100);
        assert!(
            sim < 0.5,
            "Very different sizes should have low similarity: {sim}"
        );
    }

    #[test]
    fn test_log_size_similarity_zero() {
        // Both zero → identical
        assert!((log_size_similarity(0, 0) - 1.0).abs() < 1e-10);
        // One zero, one nonzero → still produces a valid [0,1] value
        let sim = log_size_similarity(0, 10);
        assert!(sim >= 0.0 && sim <= 1.0, "Should be in [0,1]: {sim}");
    }

    #[test]
    fn test_compute_multi_signal_identical_files() {
        let source = FileSignals {
            path: "src/handlers.rs".to_string(),
            fingerprint: vec![0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.4, 0.6, 0.7, 0.3, 0.5, 0.1, 0.8, 0.2, 0.6, 0.4, 0.9],
            wl_hash: Some(12345),
            function_count: 10,
        };
        let target = source.clone();

        let result = compute_multi_signal_similarity(&source, &target);
        assert!(
            (result.similarity - 1.0).abs() < 1e-10,
            "Identical files should have similarity 1.0: {}",
            result.similarity
        );
        assert!((result.signals.fingerprint_similarity - 1.0).abs() < 1e-10);
        assert!((result.signals.wl_hash_match - 1.0).abs() < 1e-10);
        assert!((result.signals.name_similarity - 1.0).abs() < 1e-10);
        assert!((result.signals.size_similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_multi_signal_completely_different() {
        let source = FileSignals {
            path: "src/handlers.rs".to_string(),
            fingerprint: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wl_hash: Some(111),
            function_count: 1,
        };
        let target = FileSignals {
            path: "lib/models.py".to_string(),
            fingerprint: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wl_hash: Some(999),
            function_count: 100,
        };

        let result = compute_multi_signal_similarity(&source, &target);
        assert!(
            result.similarity < 0.3,
            "Completely different files should have low similarity: {}",
            result.similarity
        );
        assert!((result.signals.wl_hash_match).abs() < 1e-10); // No WL match
    }

    #[test]
    fn test_compute_multi_signal_same_role_different_project() {
        // Two "handlers.rs" files from different projects with similar structure
        let source = FileSignals {
            path: "project-a/src/api/handlers.rs".to_string(),
            fingerprint: vec![0.5, 0.3, 0.8, 0.1, 0.7, 0.2, 0.4, 0.6, 0.3, 0.2, 0.5, 0.1, 0.8, 0.2, 0.6, 0.4, 0.7],
            wl_hash: Some(555),
            function_count: 15,
        };
        let target = FileSignals {
            path: "project-b/src/api/handlers.rs".to_string(),
            fingerprint: vec![0.5, 0.35, 0.75, 0.12, 0.72, 0.18, 0.38, 0.58, 0.28, 0.22, 0.48, 0.12, 0.78, 0.22, 0.58, 0.42, 0.68],
            wl_hash: Some(555), // Same WL hash = same topology
            function_count: 18,
        };

        let result = compute_multi_signal_similarity(&source, &target);
        assert!(
            result.similarity > 0.8,
            "Same-role cross-project files should be very similar: {}",
            result.similarity
        );
        assert!((result.signals.wl_hash_match - 1.0).abs() < 1e-10); // WL match!
        assert!(result.signals.name_similarity > 0.9); // Same file name
    }

    #[test]
    fn test_compute_multi_signal_no_wl_hash() {
        let source = FileSignals {
            path: "src/main.rs".to_string(),
            fingerprint: vec![0.5; 17],
            wl_hash: None, // No WL hash available
            function_count: 5,
        };
        let target = FileSignals {
            path: "src/main.rs".to_string(),
            fingerprint: vec![0.5; 17],
            wl_hash: None,
            function_count: 5,
        };

        let result = compute_multi_signal_similarity(&source, &target);
        // WL hash should be 0.0 (missing), but other signals should compensate
        assert!((result.signals.wl_hash_match).abs() < 1e-10);
        // Fingerprint (1.0 * 0.5) + WL (0.0 * 0.2) + name (1.0 * 0.2) + size (1.0 * 0.1) = 0.8
        assert!(
            (result.similarity - 0.8).abs() < 1e-10,
            "Without WL hash: 0.5*1.0 + 0.2*0.0 + 0.2*1.0 + 0.1*1.0 = 0.8, got {}",
            result.similarity
        );
    }

    #[test]
    fn test_compute_multi_signal_empty_fingerprint() {
        let source = FileSignals {
            path: "src/main.rs".to_string(),
            fingerprint: vec![],
            wl_hash: Some(123),
            function_count: 5,
        };
        let target = FileSignals {
            path: "lib/main.rs".to_string(),
            fingerprint: vec![],
            wl_hash: Some(123),
            function_count: 5,
        };

        let result = compute_multi_signal_similarity(&source, &target);
        assert!((result.signals.fingerprint_similarity).abs() < 1e-10); // No fingerprint
        // WL match (0.2) + name match (0.2) + size match (0.1) = 0.5
        assert!(
            (result.similarity - 0.5).abs() < 0.01,
            "Without fingerprint but with matching WL + name + size: ~0.5, got {}",
            result.similarity
        );
    }

    #[test]
    fn test_find_cross_project_twins_multi_signal_ordering() {
        let source = FileSignals {
            path: "project-a/src/handlers.rs".to_string(),
            fingerprint: vec![0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.4, 0.6, 0.3, 0.2, 0.5, 0.1, 0.8, 0.2, 0.6, 0.4, 0.7],
            wl_hash: Some(100),
            function_count: 10,
        };

        let candidates = vec![
            FileSignals {
                path: "project-b/src/handlers.rs".to_string(), // Same name, same WL, similar FP
                fingerprint: vec![0.78, 0.32, 0.68, 0.12, 0.88, 0.22, 0.38, 0.58, 0.28, 0.22, 0.48, 0.12, 0.78, 0.22, 0.58, 0.42, 0.68],
                wl_hash: Some(100),
                function_count: 12,
            },
            FileSignals {
                path: "project-c/src/models.rs".to_string(), // Different name, different WL
                fingerprint: vec![0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.5, 0.8, 0.6, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2],
                wl_hash: Some(200),
                function_count: 25,
            },
            FileSignals {
                path: "project-d/src/routes.rs".to_string(), // Different name, same WL hash
                fingerprint: vec![0.75, 0.35, 0.65, 0.15, 0.85, 0.25, 0.35, 0.55, 0.35, 0.25, 0.45, 0.15, 0.75, 0.25, 0.55, 0.45, 0.65],
                wl_hash: Some(100),
                function_count: 11,
            },
        ];

        let results = find_cross_project_twins_multi_signal(&source, &candidates, 10);
        assert_eq!(results.len(), 3);

        // Should be sorted by similarity descending
        for i in 0..results.len() - 1 {
            assert!(
                results[i].similarity >= results[i + 1].similarity,
                "Results should be sorted: {} >= {}",
                results[i].similarity,
                results[i + 1].similarity
            );
        }

        // project-b/handlers.rs should be #1 (same name + same WL + similar FP)
        assert!(
            results[0].target.contains("handlers"),
            "Best match should be handlers.rs, got: {}",
            results[0].target
        );
        // project-c/models.rs should be last (different everything)
        assert!(
            results[2].target.contains("models"),
            "Worst match should be models.rs, got: {}",
            results[2].target
        );
    }

    #[test]
    fn test_find_cross_project_twins_multi_signal_top_n() {
        let source = FileSignals {
            path: "src/main.rs".to_string(),
            fingerprint: vec![0.5; 17],
            wl_hash: Some(1),
            function_count: 5,
        };

        let candidates: Vec<FileSignals> = (0..20)
            .map(|i| FileSignals {
                path: format!("project-{i}/src/file_{i}.rs"),
                fingerprint: vec![0.5 + (i as f64) * 0.01; 17],
                wl_hash: Some(i as u64),
                function_count: 5 + i,
            })
            .collect();

        let results = find_cross_project_twins_multi_signal(&source, &candidates, 5);
        assert_eq!(results.len(), 5, "Should truncate to top_n=5");
    }

    #[test]
    fn test_find_cross_project_twins_skips_self() {
        let source = FileSignals {
            path: "src/main.rs".to_string(),
            fingerprint: vec![0.5; 17],
            wl_hash: Some(1),
            function_count: 5,
        };

        // Include the source path in candidates
        let candidates = vec![
            FileSignals {
                path: "src/main.rs".to_string(), // Same path → should be skipped
                fingerprint: vec![0.5; 17],
                wl_hash: Some(1),
                function_count: 5,
            },
            FileSignals {
                path: "src/other.rs".to_string(),
                fingerprint: vec![0.4; 17],
                wl_hash: Some(2),
                function_count: 3,
            },
        ];

        let results = find_cross_project_twins_multi_signal(&source, &candidates, 10);
        assert_eq!(results.len(), 1, "Should skip self-match");
        assert_eq!(results[0].target, "src/other.rs");
    }

    #[test]
    fn test_weight_sum_is_one() {
        assert!(
            (W_FINGERPRINT + W_WL_HASH + W_NAME + W_SIZE - 1.0).abs() < 1e-10,
            "Weights should sum to 1.0"
        );
    }

    // ========================================================================
    // cluster_dna_vectors tests
    // ========================================================================

    #[test]
    fn test_cluster_dna_empty() {
        let dna: HashMap<String, Vec<f64>> = HashMap::new();
        let clusters = cluster_dna_vectors(&dna, 3);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_dna_too_few_points() {
        let mut dna = HashMap::new();
        dna.insert("a".to_string(), vec![1.0, 0.0]);
        dna.insert("b".to_string(), vec![0.0, 1.0]);
        // 2 points but 3 clusters → should return empty
        let clusters = cluster_dna_vectors(&dna, 3);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_dna_two_clear_groups() {
        // Two tight groups with distinct directions (cosine-similar within group)
        let mut dna = HashMap::new();
        // Group A: direction ~ (1, 0) — low second dimension
        dna.insert("src/handlers/auth.rs".to_string(), vec![0.9, 0.1]);
        dna.insert("src/handlers/user.rs".to_string(), vec![0.95, 0.12]);
        dna.insert("src/handlers/api.rs".to_string(), vec![0.88, 0.08]);
        // Group B: direction ~ (0, 1) — low first dimension
        dna.insert("src/models/user.rs".to_string(), vec![0.1, 0.9]);
        dna.insert("src/models/schema.rs".to_string(), vec![0.12, 0.95]);
        dna.insert("src/models/types.rs".to_string(), vec![0.08, 0.88]);

        let clusters = cluster_dna_vectors(&dna, 2);
        assert_eq!(clusters.len(), 2);

        // Each cluster should have 3 members
        let sizes: Vec<usize> = clusters.iter().map(|c| c.members.len()).collect();
        assert!(sizes.contains(&3));

        // Check that handlers are grouped together and models together
        let handler_cluster = clusters
            .iter()
            .find(|c| c.members.iter().any(|m| m.contains("handlers")))
            .unwrap();
        assert!(
            handler_cluster
                .members
                .iter()
                .all(|m| m.contains("handlers")),
            "All handler files should be in the same cluster"
        );

        // Cohesion should be high within tight clusters
        for c in &clusters {
            assert!(
                c.cohesion > 0.9,
                "Tight clusters should have high cohesion: {}",
                c.cohesion
            );
        }
    }

    #[test]
    fn test_cluster_dna_labels_inferred() {
        let mut dna = HashMap::new();
        dna.insert("src/handlers/auth_handler.rs".to_string(), vec![0.0, 0.0]);
        dna.insert("src/handlers/user_handler.rs".to_string(), vec![0.01, 0.01]);
        dna.insert("src/models/user.rs".to_string(), vec![1.0, 1.0]);
        dna.insert("src/models/schema.rs".to_string(), vec![0.99, 0.99]);

        let clusters = cluster_dna_vectors(&dna, 2);
        assert_eq!(clusters.len(), 2);

        let labels: Vec<&str> = clusters.iter().map(|c| c.label.as_str()).collect();
        // At least one should have a recognized label (not "Cluster (N)")
        assert!(
            labels.iter().any(|l| *l == "Handlers" || *l == "Models"),
            "At least one cluster should have a recognized label, got: {:?}",
            labels
        );
    }

    #[test]
    fn test_cluster_dna_single_cluster() {
        let mut dna = HashMap::new();
        dna.insert("a.rs".to_string(), vec![0.5, 0.5]);
        dna.insert("b.rs".to_string(), vec![0.6, 0.4]);
        dna.insert("c.rs".to_string(), vec![0.4, 0.6]);

        let clusters = cluster_dna_vectors(&dna, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 3);
    }

    // ====================================================================
    // Missing Link Prediction (Plan 9)
    // ====================================================================

    /// Build a chain graph: A → B → C (A and C are NOT directly connected)
    fn make_chain_abc() -> CodeGraph {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "A".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/a.rs".to_string()),
            name: "a.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "B".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/b.rs".to_string()),
            name: "b.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "C".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/c.rs".to_string()),
            name: "c.rs".to_string(),
            project_id: None,
        });
        g.add_edge(
            "A",
            "B",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "B",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g
    }

    #[test]
    fn test_find_distance_2_pairs_chain() {
        // A → B → C : (A, C) should be at distance 2
        let g = make_chain_abc();
        let pairs = find_distance_2_3_pairs(&g);
        assert_eq!(pairs.len(), 1, "Expected exactly 1 distance-2 pair");
        let pair = &pairs[0];
        let ids: HashSet<String> = [g.graph[pair.0].id.clone(), g.graph[pair.1].id.clone()]
            .into_iter()
            .collect();
        assert!(ids.contains("A"), "Pair should contain A");
        assert!(ids.contains("C"), "Pair should contain C");
    }

    #[test]
    fn test_find_distance_2_pairs_triangle_none() {
        // Triangle: A → B → C, A → C → directly connected, no distance-2 pairs
        let mut g = make_chain_abc();
        g.add_edge(
            "A",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        let pairs = find_distance_2_3_pairs(&g);
        assert!(pairs.is_empty(), "Triangle should have no distance-2 pairs");
    }

    #[test]
    fn test_find_distance_2_pairs_star_graph() {
        // Star: center → leaf_0, center → leaf_1, center → leaf_2
        // All leaves are at distance 2 from each other (through center)
        let g = make_star_graph(3);
        let pairs = find_distance_2_3_pairs(&g);
        // 3 leaves → C(3,2) = 3 pairs at distance 2
        assert_eq!(
            pairs.len(),
            3,
            "Star with 3 leaves should have 3 distance-2 pairs"
        );
    }

    #[test]
    fn test_find_distance_2_pairs_empty() {
        let g = CodeGraph::new();
        let pairs = find_distance_2_3_pairs(&g);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_distance_2_pairs_disconnected() {
        // Two disconnected nodes — no edges, no pairs
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "X".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "x.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "Y".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "y.rs".to_string(),
            project_id: None,
        });
        let pairs = find_distance_2_3_pairs(&g);
        assert!(
            pairs.is_empty(),
            "Disconnected nodes have no distance-2 pairs"
        );
    }

    #[test]
    fn test_link_plausibility_high_score() {
        // Diamond: A → B, A → C, B → D, C → D
        // B and C share common neighbors (A, D) → high Jaccard + Adamic-Adar
        let mut g = CodeGraph::new();
        for id in &["A", "B", "C", "D"] {
            g.add_node(CodeNode {
                id: id.to_string(),
                node_type: CodeNodeType::Function,
                path: Some(format!("src/{}.rs", id.to_lowercase())),
                name: id.to_string(),
                project_id: None,
            });
        }
        g.add_edge("A", "B", CodeEdge::default());
        g.add_edge("A", "C", CodeEdge::default());
        g.add_edge("B", "D", CodeEdge::default());
        g.add_edge("C", "D", CodeEdge::default());

        let b_idx = g.id_to_index["B"];
        let c_idx = g.id_to_index["C"];

        // Add co-change data
        let mut co_change = HashMap::new();
        co_change.insert(("B".to_string(), "C".to_string()), 0.8);

        let prediction = link_plausibility(&g, b_idx, c_idx, &co_change, None);

        assert!(
            prediction.plausibility > 0.4,
            "B-C with 2 common neighbors + high co-change should have plausibility > 0.4, got {}",
            prediction.plausibility
        );
        assert_eq!(prediction.signals.len(), 5, "Should have 5 signals");
        assert_eq!(prediction.suggested_relation, "CALLS"); // Function-Function
    }

    #[test]
    fn test_link_plausibility_zero_score() {
        // Two nodes far apart, no common neighbors, no co-change
        let mut g = CodeGraph::new();
        for id in &["A", "B", "C", "D", "E"] {
            g.add_node(CodeNode {
                id: id.to_string(),
                node_type: CodeNodeType::File,
                path: None,
                name: format!("{}.rs", id.to_lowercase()),
                project_id: None,
            });
        }
        // Chain: A → B → C → D → E
        g.add_edge("A", "B", CodeEdge::default());
        g.add_edge("B", "C", CodeEdge::default());
        g.add_edge("C", "D", CodeEdge::default());
        g.add_edge("D", "E", CodeEdge::default());

        let a_idx = g.id_to_index["A"];
        let e_idx = g.id_to_index["E"];
        let co_change: HashMap<(String, String), f64> = HashMap::new();

        let prediction = link_plausibility(&g, a_idx, e_idx, &co_change, None);

        // Jaccard = 0 (no common neighbors), co_change = 0, proximity = 1/4 = 0.25
        assert!(
            prediction.plausibility < 0.1,
            "A-E far apart with no co-change should have low plausibility, got {}",
            prediction.plausibility
        );
    }

    #[test]
    fn test_link_plausibility_with_dna() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "X".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "x.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "Y".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "y.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "M".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "m.rs".to_string(),
            project_id: None,
        });
        // X → M → Y (X and Y at distance 2)
        g.add_edge("X", "M", CodeEdge::default());
        g.add_edge("M", "Y", CodeEdge::default());

        let x_idx = g.id_to_index["X"];
        let y_idx = g.id_to_index["Y"];
        let co_change: HashMap<(String, String), f64> = HashMap::new();

        // Very similar DNA vectors
        let mut dna = HashMap::new();
        dna.insert("X".to_string(), vec![0.9, 0.1, 0.5]);
        dna.insert("Y".to_string(), vec![0.85, 0.15, 0.5]);

        let pred_with_dna = link_plausibility(&g, x_idx, y_idx, &co_change, Some(&dna));
        let pred_no_dna = link_plausibility(&g, x_idx, y_idx, &co_change, None);

        assert!(
            pred_with_dna.plausibility > pred_no_dna.plausibility,
            "DNA similarity should boost plausibility: with={} > without={}",
            pred_with_dna.plausibility,
            pred_no_dna.plausibility
        );
    }

    #[test]
    fn test_link_plausibility_five_signals() {
        // Verify all 5 signals are present and named correctly
        let g = make_chain_abc();
        let a_idx = g.id_to_index["A"];
        let c_idx = g.id_to_index["C"];
        let co_change: HashMap<(String, String), f64> = HashMap::new();

        let prediction = link_plausibility(&g, a_idx, c_idx, &co_change, None);
        let signal_names: Vec<&str> = prediction.signals.iter().map(|(n, _)| n.as_str()).collect();

        assert_eq!(
            signal_names,
            vec![
                "jaccard",
                "co_change",
                "proximity",
                "adamic_adar",
                "dna_similarity"
            ]
        );
    }

    #[test]
    fn test_infer_relation_type_file_file() {
        let g = make_chain_abc(); // All File nodes
        let a_idx = g.id_to_index["A"];
        let c_idx = g.id_to_index["C"];
        assert_eq!(infer_relation_type(&g, a_idx, c_idx), "IMPORTS");
    }

    #[test]
    fn test_infer_relation_type_function_function() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "fn_a".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "fn_a".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "fn_b".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "fn_b".to_string(),
            project_id: None,
        });
        let a_idx = g.id_to_index["fn_a"];
        let b_idx = g.id_to_index["fn_b"];
        assert_eq!(infer_relation_type(&g, a_idx, b_idx), "CALLS");
    }

    #[test]
    fn test_infer_relation_type_struct_trait() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "MyStruct".to_string(),
            node_type: CodeNodeType::Struct,
            path: None,
            name: "MyStruct".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "MyTrait".to_string(),
            node_type: CodeNodeType::Trait,
            path: None,
            name: "MyTrait".to_string(),
            project_id: None,
        });
        let s_idx = g.id_to_index["MyStruct"];
        let t_idx = g.id_to_index["MyTrait"];
        assert_eq!(infer_relation_type(&g, s_idx, t_idx), "IMPLEMENTS_TRAIT");
    }

    #[test]
    fn test_suggest_missing_links_co_change_boosts() {
        // A → B → C, co-change between A and C → A-C should be top prediction
        let g = make_chain_abc();
        let mut co_change = HashMap::new();
        co_change.insert(("A".to_string(), "C".to_string()), 0.9);

        let predictions = suggest_missing_links(&g, &co_change, None, 10, 0.0);
        assert_eq!(predictions.len(), 1, "Only 1 distance-2 pair: (A, C)");
        assert!(
            predictions[0].plausibility > 0.3,
            "A-C with high co-change should have plausibility > 0.3, got {}",
            predictions[0].plausibility
        );
    }

    #[test]
    fn test_suggest_missing_links_min_plausibility_filter() {
        let g = make_chain_abc();
        let co_change: HashMap<(String, String), f64> = HashMap::new();

        // With very high min_plausibility, no predictions should pass
        let predictions = suggest_missing_links(&g, &co_change, None, 10, 0.99);
        assert!(
            predictions.is_empty(),
            "No predictions should pass min_plausibility=0.99"
        );
    }

    #[test]
    fn test_suggest_missing_links_top_n_limit() {
        // Star graph: center → leaf_0, center → leaf_1, center → leaf_2
        // 3 distance-2 pairs: (leaf_0, leaf_1), (leaf_0, leaf_2), (leaf_1, leaf_2)
        let g = make_star_graph(3);
        let co_change: HashMap<(String, String), f64> = HashMap::new();

        let predictions = suggest_missing_links(&g, &co_change, None, 2, 0.0);
        assert_eq!(
            predictions.len(),
            2,
            "top_n=2 should limit to 2 predictions"
        );
    }

    #[test]
    fn test_extract_co_change_data() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "X".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "x.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "Y".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "y.rs".to_string(),
            project_id: None,
        });
        g.add_edge(
            "X",
            "Y",
            CodeEdge {
                edge_type: CodeEdgeType::CoChanged,
                weight: 0.75,
            },
        );

        let data = extract_co_change_data(&g);
        // Both directions should be present
        assert_eq!(data.get(&("X".to_string(), "Y".to_string())), Some(&0.75));
        assert_eq!(data.get(&("Y".to_string(), "X".to_string())), Some(&0.75));
    }

    #[test]
    fn test_extract_co_change_data_ignores_other_edges() {
        let g = make_chain_abc(); // Only IMPORTS edges
        let data = extract_co_change_data(&g);
        assert!(
            data.is_empty(),
            "IMPORTS edges should not be extracted as co-change"
        );
    }

    // ========================================================================
    // Stress Testing (Plan 5) tests
    // ========================================================================

    /// Helper: build a star graph (center + N leaves).
    /// All edges are center → leaf_i.
    fn make_star_stress(n: usize) -> CodeGraph {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "center".to_string(),
            node_type: CodeNodeType::File,
            path: Some("center".to_string()),
            name: "center".to_string(),
            project_id: None,
        });
        for i in 0..n {
            let id = format!("leaf_{}", i);
            g.add_node(CodeNode {
                id: id.clone(),
                node_type: CodeNodeType::File,
                path: Some(id.clone()),
                name: id.clone(),
                project_id: None,
            });
            g.add_edge(
                "center",
                &id,
                CodeEdge {
                    edge_type: CodeEdgeType::Imports,
                    weight: 1.0,
                },
            );
        }
        g
    }

    /// Helper: A-B-C-D chain.
    fn make_chain_stress() -> CodeGraph {
        let mut g = CodeGraph::new();
        for id in &["A", "B", "C", "D"] {
            g.add_node(CodeNode {
                id: id.to_string(),
                node_type: CodeNodeType::File,
                path: Some(id.to_string()),
                name: id.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "A",
            "B",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "B",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "C",
            "D",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g
    }

    // -- stress_test_node_removal --

    #[test]
    fn test_stress_node_removal_star_center() {
        let g = make_star_stress(4);
        let result = stress_test_node_removal(&g, "center").unwrap();
        assert_eq!(
            result.mode,
            super::super::models::StressTestMode::NodeRemoval
        );
        assert_eq!(result.target, "center");
        // Removing the center should orphan all 4 leaves
        assert_eq!(result.orphaned_nodes, 4);
        assert_eq!(result.components_before, 1);
        assert_eq!(result.components_after, 4);
        assert!(
            result.resilience_score < 0.1,
            "Resilience should be very low"
        );
    }

    #[test]
    fn test_stress_node_removal_leaf() {
        let g = make_star_stress(4);
        let result = stress_test_node_removal(&g, "leaf_0").unwrap();
        // Removing a leaf should have minimal impact
        assert_eq!(result.orphaned_nodes, 0);
        assert_eq!(result.components_before, 1);
        assert_eq!(result.components_after, 1);
        assert!(result.resilience_score > 0.9, "Resilience should be high");
    }

    #[test]
    fn test_stress_node_removal_chain_middle() {
        let g = make_chain_stress(); // A-B-C-D
        let result = stress_test_node_removal(&g, "B").unwrap();
        // Removing B splits: A alone, C-D together
        assert_eq!(result.components_before, 1);
        assert_eq!(result.components_after, 2);
        assert_eq!(result.orphaned_nodes, 1); // A becomes singleton
    }

    #[test]
    fn test_stress_node_removal_unknown() {
        let g = make_chain_stress();
        assert!(stress_test_node_removal(&g, "nonexistent").is_none());
    }

    #[test]
    fn test_stress_node_removal_single_node() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "alone".to_string(),
            node_type: CodeNodeType::File,
            path: Some("alone".to_string()),
            name: "alone".to_string(),
            project_id: None,
        });
        let result = stress_test_node_removal(&g, "alone").unwrap();
        assert_eq!(result.resilience_score, 0.0);
        assert_eq!(result.components_before, 1);
        assert_eq!(result.components_after, 0);
    }

    // -- find_bridges --

    #[test]
    fn test_find_bridges_chain() {
        let g = make_chain_stress(); // A-B-C-D
        let bridges = find_bridges(&g);
        // In a chain, every edge is a bridge
        assert_eq!(bridges.len(), 3, "Chain of 4 should have 3 bridges");
    }

    #[test]
    fn test_find_bridges_cycle() {
        // A-B-C-A: no bridges (cycle)
        let mut g = CodeGraph::new();
        for id in &["A", "B", "C"] {
            g.add_node(CodeNode {
                id: id.to_string(),
                node_type: CodeNodeType::File,
                path: Some(id.to_string()),
                name: id.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "A",
            "B",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "B",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "C",
            "A",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        let bridges = find_bridges(&g);
        assert!(bridges.is_empty(), "Cycle should have no bridges");
    }

    #[test]
    fn test_find_bridges_star() {
        let g = make_star_stress(3);
        let bridges = find_bridges(&g);
        // Star: every edge is a bridge
        assert_eq!(bridges.len(), 3, "Star with 3 leaves should have 3 bridges");
    }

    #[test]
    fn test_find_bridges_empty() {
        let g = CodeGraph::new();
        let bridges = find_bridges(&g);
        assert!(bridges.is_empty());
    }

    // -- stress_test_cascade --

    #[test]
    fn test_stress_cascade_chain() {
        // A→B→C→D: removing A should cascade: A removed → B has no incoming
        // but depends on direction. In cascade, we remove nodes with NO remaining incoming edges.
        let g = make_chain_stress(); // A→B→C→D
        let result = stress_test_cascade(&g, "A", 10).unwrap();
        assert_eq!(result.mode, super::super::models::StressTestMode::Cascade);
        // A removed → B loses its only incoming → B removed → C removed → D removed
        assert_eq!(
            result.blast_radius, 4,
            "Full cascade should remove all 4 nodes"
        );
        assert!(
            result.cascade_depth >= 1,
            "Should have at least 1 cascade round"
        );
    }

    #[test]
    fn test_stress_cascade_star_center() {
        let g = make_star_stress(3); // center→leaf_0, center→leaf_1, center→leaf_2
        let result = stress_test_cascade(&g, "center", 10).unwrap();
        // center removed → all leaves lose incoming → all removed
        assert_eq!(result.blast_radius, 4); // center + 3 leaves
    }

    #[test]
    fn test_stress_cascade_leaf() {
        let g = make_star_stress(3);
        let result = stress_test_cascade(&g, "leaf_0", 10).unwrap();
        // Removing a leaf shouldn't cascade
        assert_eq!(result.blast_radius, 1); // only the leaf itself
        assert_eq!(result.cascade_depth, 0);
    }

    #[test]
    fn test_stress_cascade_unknown() {
        let g = make_chain_stress();
        assert!(stress_test_cascade(&g, "nonexistent", 10).is_none());
    }

    #[test]
    fn test_stress_cascade_max_iterations() {
        let g = make_chain_stress(); // A→B→C→D
                                     // With max_iterations=1, cascade should stop after 1 round
        let result = stress_test_cascade(&g, "A", 1).unwrap();
        assert!(result.cascade_depth <= 1);
    }

    // -- stress_test_edge_removal --

    #[test]
    fn test_stress_edge_removal_bridge() {
        let g = make_chain_stress(); // A-B-C-D
        let result = stress_test_edge_removal(&g, "B", "C").unwrap();
        assert_eq!(
            result.mode,
            super::super::models::StressTestMode::EdgeRemoval
        );
        // B-C is a bridge in the chain
        assert_eq!(
            result.resilience_score, 0.0,
            "Bridge removal should give 0 resilience"
        );
        assert!(result.components_after > result.components_before);
        assert_eq!(result.critical_edges.len(), 1);
    }

    #[test]
    fn test_stress_edge_removal_non_bridge() {
        // A-B-C-A cycle: no edge is a bridge
        let mut g = CodeGraph::new();
        for id in &["A", "B", "C"] {
            g.add_node(CodeNode {
                id: id.to_string(),
                node_type: CodeNodeType::File,
                path: Some(id.to_string()),
                name: id.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "A",
            "B",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "B",
            "C",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "C",
            "A",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        let result = stress_test_edge_removal(&g, "A", "B").unwrap();
        assert_eq!(
            result.resilience_score, 1.0,
            "Non-bridge should give 1.0 resilience"
        );
        assert_eq!(result.components_before, result.components_after);
        assert!(result.critical_edges.is_empty());
    }

    #[test]
    fn test_stress_edge_removal_unknown() {
        let g = make_chain_stress();
        assert!(stress_test_edge_removal(&g, "A", "nonexistent").is_none());
    }

    // ========================================================================
    // Context Cards (Plan 8) tests
    // ========================================================================

    /// Helper: 5-file graph for context card tests.
    fn make_file_graph_cc() -> CodeGraph {
        let mut g = CodeGraph::new();
        let files = [
            "src/main.rs",
            "src/lib.rs",
            "src/api/mod.rs",
            "src/api/handlers.rs",
            "src/api/routes.rs",
        ];
        for path in &files {
            g.add_node(CodeNode {
                id: path.to_string(),
                node_type: CodeNodeType::File,
                path: Some(path.to_string()),
                name: path.rsplit('/').next().unwrap_or(path).to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "src/main.rs",
            "src/lib.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/main.rs",
            "src/api/mod.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/mod.rs",
            "src/api/handlers.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/mod.rs",
            "src/api/routes.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/routes.rs",
            "src/api/handlers.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g
    }

    #[test]
    fn test_context_cards_file_graph() {
        use super::super::models::AnalyticsConfig;
        // 5-file graph: main → lib, main → api/mod, api/mod → handlers, api/mod → routes, routes → handlers
        let g = make_file_graph_cc();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);
        let dna_map = HashMap::new();
        let wl_hashes = HashMap::new();

        let cards = compute_context_cards(&g, &analytics, &dna_map, &wl_hashes, &HashMap::new());

        assert_eq!(cards.len(), 5, "Should produce 1 card per file node");

        // All cards should have positive pagerank (connected graph)
        for card in &cards {
            assert!(
                card.cc_pagerank > 0.0,
                "PageRank should be > 0 for {}",
                card.path
            );
            assert_eq!(card.cc_version, 1);
            assert!(!card.cc_computed_at.is_empty());
        }

        // Check main.rs has 2 outgoing imports (lib + api/mod) and 0 incoming
        let main_card = cards.iter().find(|c| c.path == "src/main.rs").unwrap();
        assert_eq!(main_card.cc_imports_out, 2, "main.rs should import 2 files");
        assert_eq!(
            main_card.cc_imports_in, 0,
            "main.rs should have 0 importers"
        );

        // Check handlers has 0 outgoing and 2 incoming (api/mod + routes)
        let handlers_card = cards
            .iter()
            .find(|c| c.path == "src/api/handlers.rs")
            .unwrap();
        assert_eq!(handlers_card.cc_imports_out, 0);
        assert_eq!(handlers_card.cc_imports_in, 2);
    }

    #[test]
    fn test_context_cards_only_files() {
        // Mix of File + Function nodes: only File nodes get cards
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/main.rs".to_string()),
            name: "main.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "my_func".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "my_func".to_string(),
            project_id: None,
        });
        g.add_edge(
            "src/main.rs",
            "my_func",
            CodeEdge {
                edge_type: CodeEdgeType::Defines,
                weight: 1.0,
            },
        );

        use super::super::models::AnalyticsConfig;
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);
        let cards = compute_context_cards(&g, &analytics, &HashMap::new(), &HashMap::new(), &HashMap::new());

        assert_eq!(cards.len(), 1, "Only File nodes should get context cards");
        assert_eq!(cards[0].path, "src/main.rs");
    }

    #[test]
    fn test_context_cards_with_dna_and_wl() {
        let g = make_chain_abc(); // A→B→C
        use super::super::models::AnalyticsConfig;
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        let mut dna_map = HashMap::new();
        dna_map.insert("A".to_string(), vec![1.0, 0.5, 0.0]);
        let mut wl_hashes = HashMap::new();
        wl_hashes.insert("A".to_string(), 12345u64);

        let cards = compute_context_cards(&g, &analytics, &dna_map, &wl_hashes, &HashMap::new());
        let a_card = cards.iter().find(|c| c.path == "A").unwrap();
        assert_eq!(a_card.cc_structural_dna, vec![1.0, 0.5, 0.0]);
        assert_eq!(a_card.cc_wl_hash, 12345);

        // B should have no DNA or WL
        let b_card = cards.iter().find(|c| c.path == "B").unwrap();
        assert!(b_card.cc_structural_dna.is_empty());
        assert_eq!(b_card.cc_wl_hash, 0);
    }

    #[test]
    fn test_context_cards_empty_graph() {
        use super::super::models::AnalyticsConfig;
        let g = CodeGraph::new();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);
        let cards = compute_context_cards(&g, &analytics, &HashMap::new(), &HashMap::new(), &HashMap::new());
        assert!(cards.is_empty());
    }

    #[test]
    fn test_context_cards_default() {
        use super::super::models::ContextCard;
        let card = ContextCard::default();
        assert_eq!(card.cc_version, 0);
        assert_eq!(card.cc_pagerank, 0.0);
        assert!(card.path.is_empty());
    }

    // ── WL Subgraph Hash tests ──────────────────────────────────────

    #[test]
    fn test_wl_node_hash_single_node() {
        let mut g = CodeGraph::new();
        g.graph.add_node(CodeNode {
            id: "A".into(),
            node_type: CodeNodeType::Function,
            name: "a".into(),
            path: None,
            project_id: None,
        });
        let idx = g.graph.node_indices().next().unwrap();
        let hash = wl_node_hash(&g, idx, 2, 3);
        assert_ne!(hash, 0, "Hash of a single node should be non-zero");
    }

    #[test]
    fn test_wl_node_hash_deterministic() {
        let g = make_chain_graph(5);
        let idx = g.graph.node_indices().next().unwrap();
        let h1 = wl_node_hash(&g, idx, 2, 3);
        let h2 = wl_node_hash(&g, idx, 2, 3);
        assert_eq!(h1, h2, "WL hash should be deterministic");
    }

    #[test]
    fn test_wl_node_hash_symmetric_nodes_same_hash() {
        // Star graph: center + leaves. All leaves should have the same hash.
        let g = make_star_graph(4);
        let leaf_indices: Vec<_> = g
            .graph
            .node_indices()
            .filter(|&i| g.graph[i].id != "center")
            .collect();
        let hashes: Vec<u64> = leaf_indices
            .iter()
            .map(|&i| wl_node_hash(&g, i, 2, 3))
            .collect();
        assert!(
            hashes.windows(2).all(|w| w[0] == w[1]),
            "Symmetric leaf nodes in star graph should have identical WL hashes"
        );
    }

    #[test]
    fn test_wl_node_hash_different_positions_differ() {
        // In a chain A-B-C-D-E, endpoint A and center C should have different hashes
        let g = make_chain_graph(5);
        let indices: Vec<_> = g.graph.node_indices().collect();
        let hash_first = wl_node_hash(&g, indices[0], 2, 3);
        let hash_middle = wl_node_hash(&g, indices[2], 2, 3);
        assert_ne!(
            hash_first, hash_middle,
            "Endpoint and center of a chain should have different WL hashes"
        );
    }

    #[test]
    fn test_wl_node_hash_radius_affects_result() {
        let g = make_chain_graph(10);
        let idx = g.graph.node_indices().nth(5).unwrap();
        let h_r1 = wl_node_hash(&g, idx, 1, 3);
        let h_r3 = wl_node_hash(&g, idx, 3, 3);
        // Different radius = different neighborhood = different hash
        // (unless the graph is trivially small, which 10 nodes is not)
        assert_ne!(
            h_r1, h_r3,
            "Different radii should generally produce different hashes"
        );
    }

    #[test]
    fn test_wl_subgraph_hash_all_basic() {
        let g = make_chain_graph(5);
        let hashes = wl_subgraph_hash_all(&g, 2, 3).unwrap();
        assert_eq!(hashes.len(), 5, "Should have a hash for every node");
        for hash in hashes.values() {
            assert_ne!(*hash, 0, "No hash should be zero");
        }
    }

    #[test]
    fn test_wl_subgraph_hash_all_empty_graph() {
        let g = CodeGraph::new();
        let hashes = wl_subgraph_hash_all(&g, 2, 3).unwrap();
        assert!(hashes.is_empty(), "Empty graph should produce no hashes");
    }

    #[test]
    fn test_wl_subgraph_hash_all_star_leaves_same() {
        let g = make_star_graph(5);
        // radius=1 so center sees all leaves but each leaf only sees center
        let hashes = wl_subgraph_hash_all(&g, 1, 3).unwrap();
        let leaf_hashes: Vec<u64> = hashes
            .iter()
            .filter(|(k, _)| *k != "center")
            .map(|(_, &v)| v)
            .collect();
        assert!(
            leaf_hashes.windows(2).all(|w| w[0] == w[1]),
            "All leaf hashes in a star should be identical"
        );
        // Center should differ from leaves (different neighborhood structure at radius=1)
        let center_hash = hashes["center"];
        assert_ne!(
            center_hash, leaf_hashes[0],
            "Center hash should differ from leaf hash"
        );
    }

    // ── find_isomorphic_groups tests ────────────────────────────────

    #[test]
    fn test_find_isomorphic_groups_basic() {
        let mut hashes = HashMap::new();
        hashes.insert("A".to_string(), 100u64);
        hashes.insert("B".to_string(), 100u64);
        hashes.insert("C".to_string(), 200u64);
        hashes.insert("D".to_string(), 200u64);
        hashes.insert("E".to_string(), 300u64);

        let groups = find_isomorphic_groups(&hashes);
        assert_eq!(
            groups.len(),
            2,
            "Should have 2 groups (singletons excluded)"
        );
        assert_eq!(groups[&100].len(), 2);
        assert_eq!(groups[&200].len(), 2);
    }

    #[test]
    fn test_find_isomorphic_groups_no_duplicates() {
        let mut hashes = HashMap::new();
        hashes.insert("A".to_string(), 1u64);
        hashes.insert("B".to_string(), 2u64);
        hashes.insert("C".to_string(), 3u64);

        let groups = find_isomorphic_groups(&hashes);
        assert!(
            groups.is_empty(),
            "All unique hashes should produce no groups"
        );
    }

    #[test]
    fn test_find_isomorphic_groups_all_same() {
        let mut hashes = HashMap::new();
        for i in 0..5 {
            hashes.insert(format!("node_{}", i), 42u64);
        }
        let groups = find_isomorphic_groups(&hashes);
        assert_eq!(groups.len(), 1, "All same hash → one group");
        assert_eq!(groups[&42].len(), 5);
    }

    #[test]
    fn test_find_isomorphic_groups_empty() {
        let hashes: HashMap<String, u64> = HashMap::new();
        let groups = find_isomorphic_groups(&hashes);
        assert!(groups.is_empty());
    }

    // ── double_radius_label tests ───────────────────────────────────

    #[test]
    fn test_double_radius_label_linear() {
        // A → B → C, source=A, target=C
        let paths: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];
        let labels = double_radius_label(&paths, &edges, "A", "C");
        assert_eq!(labels["A"], (0, 2)); // dist(A,A)=0, dist(A,C)=2
        assert_eq!(labels["B"], (1, 1)); // dist(B,A)=1, dist(B,C)=1
        assert_eq!(labels["C"], (2, 0)); // dist(C,A)=2, dist(C,C)=0
    }

    #[test]
    fn test_double_radius_label_star() {
        // Center connected to A, B, C. source=A, target=B
        let paths: Vec<String> = vec!["center".into(), "A".into(), "B".into(), "C".into()];
        let edges = vec![
            ("center".to_string(), "A".to_string()),
            ("center".to_string(), "B".to_string()),
            ("center".to_string(), "C".to_string()),
        ];
        let labels = double_radius_label(&paths, &edges, "A", "B");
        assert_eq!(labels["A"], (0, 2));
        assert_eq!(labels["B"], (2, 0));
        assert_eq!(labels["center"], (1, 1));
        assert_eq!(labels["C"], (2, 2)); // equidistant
    }

    #[test]
    fn test_double_radius_label_disconnected() {
        // A-B and C (disconnected), source=A, target=C
        let paths: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let edges = vec![("A".to_string(), "B".to_string())];
        let labels = double_radius_label(&paths, &edges, "A", "C");
        assert_eq!(labels["A"].0, 0); // dist to source = 0
        assert_eq!(labels["A"].1, u32::MAX); // unreachable to target
        assert_eq!(labels["C"].0, u32::MAX); // unreachable from source
        assert_eq!(labels["C"].1, 0); // dist to target = 0
    }

    // ── find_bottleneck_nodes tests ─────────────────────────────────

    #[test]
    fn test_bottleneck_chain() {
        // A - B - C - D - E, source=A, target=E
        let paths: Vec<String> = (0..5).map(|i| format!("n{}", i)).collect();
        let edges: Vec<(String, String)> = (0..4)
            .map(|i| (format!("n{}", i), format!("n{}", i + 1)))
            .collect();
        let bottlenecks = find_bottleneck_nodes(&paths, &edges, "n0", "n4", 3);
        // Middle node n2 should be highest betweenness
        assert!(
            !bottlenecks.is_empty(),
            "Chain should have bottleneck nodes"
        );
        assert_eq!(
            bottlenecks[0], "n2",
            "Center of chain should be top bottleneck"
        );
    }

    #[test]
    fn test_bottleneck_too_small() {
        // Only 2 nodes: source + target, no intermediates
        let paths = vec!["A".to_string(), "B".to_string()];
        let edges = vec![("A".to_string(), "B".to_string())];
        let bottlenecks = find_bottleneck_nodes(&paths, &edges, "A", "B", 5);
        assert!(bottlenecks.is_empty(), "2 nodes = no intermediates");
    }

    #[test]
    fn test_bottleneck_star_center() {
        // Star: center - A, center - B, center - C, center - D
        // source=A, target=B → center is the bottleneck
        let paths: Vec<String> = vec![
            "center".into(),
            "A".into(),
            "B".into(),
            "C".into(),
            "D".into(),
        ];
        let edges: Vec<(String, String)> = vec![
            ("center".into(), "A".into()),
            ("center".into(), "B".into()),
            ("center".into(), "C".into()),
            ("center".into(), "D".into()),
        ];
        let bottlenecks = find_bottleneck_nodes(&paths, &edges, "A", "B", 1);
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0], "center");
    }

    #[test]
    fn test_bottleneck_excludes_source_target() {
        let paths: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];
        let bottlenecks = find_bottleneck_nodes(&paths, &edges, "A", "C", 10);
        // Should not contain A or C
        assert!(!bottlenecks.contains(&"A".to_string()));
        assert!(!bottlenecks.contains(&"C".to_string()));
    }

    // ── find_distance_2_3_pairs tests ───────────────────────────────

    #[test]
    fn test_distance_2_3_pairs_chain() {
        // Chain A-B-C-D: distance-2 pairs = (A,C) and (B,D)
        let g = make_chain_graph(4);
        let pairs = find_distance_2_3_pairs(&g);
        assert!(!pairs.is_empty(), "Chain of 4 should have distance-2 pairs");
        // A-C and B-D are at distance 2 (not directly connected)
        // A-D is distance 3, NOT captured by 2-hop BFS
        assert_eq!(pairs.len(), 2, "A-C and B-D at distance 2");
    }

    #[test]
    fn test_distance_2_3_pairs_complete_graph() {
        // In a complete graph (clique), every node is directly connected,
        // so there are no distance-2 pairs
        let g = make_two_cliques(3); // creates 2 cliques connected by a bridge
        let pairs = find_distance_2_3_pairs(&g);
        // Two cliques connected → there will be cross-clique distance-2 pairs
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_distance_2_3_pairs_empty() {
        let g = CodeGraph::new();
        let pairs = find_distance_2_3_pairs(&g);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_distance_2_3_pairs_star() {
        // Star: center - L1, center - L2, center - L3
        // All leaves are distance 2 from each other (through center)
        let g = make_star_graph(4);
        let pairs = find_distance_2_3_pairs(&g);
        // 4 leaves → C(4,2) = 6 distance-2 pairs
        assert_eq!(
            pairs.len(),
            6,
            "Star with 4 leaves: 6 leaf-leaf pairs at distance 2"
        );
    }

    // ========================================================================
    // Structural Fingerprint v2 tests
    // ========================================================================

    /// Build a file-centric graph for fingerprint testing.
    /// Creates File nodes with Defines edges to Function/Struct children,
    /// and Import edges between files.
    fn make_fingerprint_test_graph() -> (CodeGraph, GraphAnalytics) {
        let mut g = CodeGraph::new();

        // Hub file: imported by many, defines many functions
        g.add_node(CodeNode {
            id: "hub.rs".into(),
            node_type: CodeNodeType::File,
            path: Some("hub.rs".into()),
            name: "hub.rs".into(),
            project_id: None,
        });
        for i in 0..5 {
            let fname = format!("hub_fn_{}", i);
            g.add_node(CodeNode {
                id: fname.clone(),
                node_type: CodeNodeType::Function,
                path: Some("hub.rs".into()),
                name: fname.clone(),
                project_id: None,
            });
            g.add_edge("hub.rs", &fname, CodeEdge {
                edge_type: CodeEdgeType::Defines,
                weight: 1.0,
            });
        }
        // Hub also defines a struct
        g.add_node(CodeNode {
            id: "HubStruct".into(),
            node_type: CodeNodeType::Struct,
            path: Some("hub.rs".into()),
            name: "HubStruct".into(),
            project_id: None,
        });
        g.add_edge("hub.rs", "HubStruct", CodeEdge {
            edge_type: CodeEdgeType::Defines,
            weight: 1.0,
        });

        // 5 leaf files: each imports hub
        for i in 0..5 {
            let leaf = format!("leaf_{}.rs", i);
            g.add_node(CodeNode {
                id: leaf.clone(),
                node_type: CodeNodeType::File,
                path: Some(leaf.clone()),
                name: leaf.clone(),
                project_id: None,
            });
            g.add_edge(&leaf, "hub.rs", CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            });
            // Each leaf defines 1 function
            let fn_name = format!("leaf_fn_{}", i);
            g.add_node(CodeNode {
                id: fn_name.clone(),
                node_type: CodeNodeType::Function,
                path: Some(leaf.clone()),
                name: fn_name.clone(),
                project_id: None,
            });
            g.add_edge(&leaf, &fn_name, CodeEdge {
                edge_type: CodeEdgeType::Defines,
                weight: 1.0,
            });
        }

        // Peripheral file: no imports, 1 function
        g.add_node(CodeNode {
            id: "orphan.rs".into(),
            node_type: CodeNodeType::File,
            path: Some("orphan.rs".into()),
            name: "orphan.rs".into(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "orphan_fn".into(),
            node_type: CodeNodeType::Function,
            path: Some("orphan.rs".into()),
            name: "orphan_fn".into(),
            project_id: None,
        });
        g.add_edge("orphan.rs", "orphan_fn", CodeEdge {
            edge_type: CodeEdgeType::Defines,
            weight: 1.0,
        });

        // Compute analytics
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&g, &config);

        (g, analytics)
    }

    #[test]
    fn test_fingerprint_empty_graph() {
        let g = CodeGraph::new();
        let analytics = GraphAnalytics {
            metrics: HashMap::new(),
            communities: Vec::new(),
            components: Vec::new(),
            health: Default::default(),
            modularity: 0.0,
            node_count: 0,
            edge_count: 0,
            computation_ms: 0,
            profile_name: None,
        };
        let fp = compute_structural_fingerprint(&g, &analytics);
        assert!(fp.is_empty(), "Empty graph should produce empty fingerprint map");
    }

    #[test]
    fn test_fingerprint_dimensions_and_range() {
        use crate::graph::models::{FINGERPRINT_DIMS, FINGERPRINT_LABELS};

        let (g, analytics) = make_fingerprint_test_graph();
        let fp = compute_structural_fingerprint(&g, &analytics);

        // Should have fingerprints for all 7 File nodes (hub + 5 leaves + orphan)
        assert_eq!(fp.len(), 7, "Should have 7 file fingerprints");

        // Each fingerprint should have 17 dimensions, all in [0, 1]
        for (path, vec) in &fp {
            assert_eq!(
                vec.len(),
                FINGERPRINT_DIMS,
                "File {} should have {} dims, got {}",
                path,
                FINGERPRINT_DIMS,
                vec.len()
            );
            for (dim_idx, &val) in vec.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "File {} dim {} ({}) = {} out of [0,1]",
                    path,
                    dim_idx,
                    FINGERPRINT_LABELS[dim_idx],
                    val
                );
            }
        }
    }

    #[test]
    fn test_fingerprint_hub_vs_leaf_discrimination() {
        let (g, analytics) = make_fingerprint_test_graph();
        let fp = compute_structural_fingerprint(&g, &analytics);

        let hub = fp.get("hub.rs").expect("hub should have fingerprint");
        let leaf = fp.get("leaf_0.rs").expect("leaf should have fingerprint");
        let orphan = fp.get("orphan.rs").expect("orphan should have fingerprint");

        // Hub should have HIGH imports_in percentile (d0) — it's imported by 5 files
        assert!(
            hub[0] > leaf[0],
            "Hub imports_in_pct ({}) should be > leaf ({}) — hub is imported by 5 files",
            hub[0], leaf[0]
        );

        // Hub should have HIGH function_count percentile (d7) — 5 functions vs 1
        assert!(
            hub[7] > leaf[7],
            "Hub function_count_pct ({}) should be > leaf ({})",
            hub[7], leaf[7]
        );

        // Hub should be a hub (d14 = 0.0) or at least not peripheral
        // Orphan should be peripheral (d14 = 1.0)
        assert!(
            hub[14] < orphan[14],
            "Hub community_role ({}) should be < orphan ({}) — hub is more central",
            hub[14], orphan[14]
        );

        // Cosine similarity between hub and leaf should be < 0.85
        // (proves discrimination between roles)
        let sim = cosine_similarity(hub, leaf);
        assert!(
            sim < 0.85,
            "Hub-leaf cosine similarity ({}) should be < 0.85 — different roles",
            sim
        );
    }

    #[test]
    fn test_fingerprint_symmetric_leaves_similar() {
        let (g, analytics) = make_fingerprint_test_graph();
        let fp = compute_structural_fingerprint(&g, &analytics);

        let leaf0 = fp.get("leaf_0.rs").unwrap();
        let leaf1 = fp.get("leaf_1.rs").unwrap();

        // Two leaves with identical structure should have very similar fingerprints
        // (not perfect 1.0 because percentile ranking may break ties differently)
        let sim = cosine_similarity(leaf0, leaf1);
        assert!(
            sim > 0.90,
            "Symmetric leaves should have cosine > 0.90, got {}",
            sim
        );
    }

    #[test]
    fn test_log_percentile_power_law_spread() {
        // Simulate a power-law distribution (many small, few large)
        let values = vec![1.0, 1.0, 2.0, 5.0, 20.0, 100.0];

        let log_pct = to_log_percentiles(&values);
        let lin_pct = to_linear_percentiles(&values);

        // Log-percentile design: spread low values MORE, compress high values.
        // This is exactly what we want for power-law metrics (degree, pagerank)
        // where discrimination among low values matters most.

        // Bottom pair (1.0 vs 2.0 = indices 1 vs 2): log should spread MORE
        let log_low_spread = log_pct[2] - log_pct[1]; // 2.0 vs 1.0
        let lin_low_spread = lin_pct[2] - lin_pct[1]; // 2.0 vs 1.0
        assert!(
            log_low_spread > lin_low_spread,
            "Log percentile should spread low values MORE than linear: log={}, lin={}",
            log_low_spread, lin_low_spread
        );

        // Top pair (20 vs 100 = indices 4 vs 5): log should COMPRESS
        let log_high_spread = log_pct[5] - log_pct[4]; // 100 vs 20
        let lin_high_spread = lin_pct[5] - lin_pct[4]; // 100 vs 20
        assert!(
            log_high_spread < lin_high_spread,
            "Log percentile should compress high values: log={}, lin={}",
            log_high_spread, lin_high_spread
        );

        // All values should be in [0, 1]
        for &v in &log_pct {
            assert!((0.0..=1.0).contains(&v), "Log percentile {} out of range", v);
        }
        for &v in &lin_pct {
            assert!((0.0..=1.0).contains(&v), "Lin percentile {} out of range", v);
        }
    }

    #[test]
    fn test_normalized_shannon_entropy() {
        // Uniform distribution → max entropy = 1.0
        let uniform = vec![10, 10, 10, 10];
        let h = normalized_shannon_entropy(&uniform);
        assert!(
            (h - 1.0).abs() < 0.01,
            "Uniform distribution should have entropy ~1.0, got {}",
            h
        );

        // Single category → zero entropy
        let single = vec![42, 0, 0, 0];
        let h = normalized_shannon_entropy(&single);
        assert!(
            h.abs() < 0.01,
            "Single category should have entropy ~0.0, got {}",
            h
        );

        // Empty → zero
        let empty: Vec<usize> = vec![];
        assert_eq!(normalized_shannon_entropy(&empty), 0.0);
    }
}
