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
    if wl_hashes.is_empty() {
        tracing::warn!("context_cards will set cc_wl_hash = None — WL hash unavailable");
    }
    let (_cards_result, cards_timing) = timed_step("context_cards", || {
        // TODO(Plan 8): compute_context_cards(&analytics, &structural_dna_map, &wl_hashes, ...)
        // When wl_hashes is empty, cards set cc_wl_hash = None
        grail_stats.cards_computed = 0;
        Ok(())
    });
    timings.push(cards_timing);

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
        predicted_links,
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

        for ci in 0..n_clusters {
            if counts[ci] > 0 {
                for d in 0..dim {
                    new_centroids[ci][d] /= counts[ci] as f64;
                }
            }
        }

        centroids = new_centroids;
    }

    // --- Build clusters ---
    let mut clusters: Vec<super::models::DnaCluster> = Vec::with_capacity(n_clusters);

    for ci in 0..n_clusters {
        let member_indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &a)| a == ci)
            .map(|(i, _)| i)
            .collect();

        if member_indices.is_empty() {
            continue; // Skip empty clusters
        }

        let members: Vec<String> = member_indices
            .iter()
            .map(|&i| paths[i].clone())
            .collect();

        // Compute intra-cluster cohesion (average pairwise cosine similarity)
        let cohesion = if members.len() <= 1 {
            1.0
        } else {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..member_indices.len() {
                for j in (i + 1)..member_indices.len() {
                    sum += cosine_similarity(vectors[member_indices[i]], vectors[member_indices[j]]);
                    count += 1;
                }
            }
            if count > 0 { sum / count as f64 } else { 1.0 }
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
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
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
        let neighbors: HashSet<NodeIndex> =
            undirected_neighbors(g, node).into_iter().collect();

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
pub fn infer_relation_type(
    graph: &CodeGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> String {
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
                assert!(d >= 0.0 && d <= 1.0, "DNA value out of range: {}", d);
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
            handler_cluster.members.iter().all(|m| m.contains("handlers")),
            "All handler files should be in the same cluster"
        );

        // Cohesion should be high within tight clusters
        for c in &clusters {
            assert!(c.cohesion > 0.9, "Tight clusters should have high cohesion: {}", c.cohesion);
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
        g.add_edge("A", "B", CodeEdge {
            edge_type: CodeEdgeType::Imports,
            weight: 1.0,
        });
        g.add_edge("B", "C", CodeEdge {
            edge_type: CodeEdgeType::Imports,
            weight: 1.0,
        });
        g
    }

    #[test]
    fn test_find_distance_2_pairs_chain() {
        // A → B → C : (A, C) should be at distance 2
        let g = make_chain_abc();
        let pairs = find_distance_2_3_pairs(&g);
        assert_eq!(pairs.len(), 1, "Expected exactly 1 distance-2 pair");
        let pair = &pairs[0];
        let ids: HashSet<String> = [
            g.graph[pair.0].id.clone(),
            g.graph[pair.1].id.clone(),
        ]
        .into_iter()
        .collect();
        assert!(ids.contains("A"), "Pair should contain A");
        assert!(ids.contains("C"), "Pair should contain C");
    }

    #[test]
    fn test_find_distance_2_pairs_triangle_none() {
        // Triangle: A → B → C, A → C → directly connected, no distance-2 pairs
        let mut g = make_chain_abc();
        g.add_edge("A", "C", CodeEdge {
            edge_type: CodeEdgeType::Imports,
            weight: 1.0,
        });
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
        assert_eq!(pairs.len(), 3, "Star with 3 leaves should have 3 distance-2 pairs");
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
        assert!(pairs.is_empty(), "Disconnected nodes have no distance-2 pairs");
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

        assert_eq!(signal_names, vec![
            "jaccard", "co_change", "proximity", "adamic_adar", "dna_similarity"
        ]);
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
        assert_eq!(predictions.len(), 2, "top_n=2 should limit to 2 predictions");
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
        assert!(data.is_empty(), "IMPORTS edges should not be extracted as co-change");
    }
}
