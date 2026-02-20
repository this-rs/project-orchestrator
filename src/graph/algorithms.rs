//! Graph analytics algorithms.
//!
//! Implements core graph data science algorithms on petgraph graphs:
//! - **PageRank** — power iteration (custom implementation)
//! - **Betweenness centrality** — via `rustworkx_core::centrality::betweenness_centrality`
//! - **Community detection (Louvain)** — custom implementation
//! - **Clustering coefficient** — local clustering per node
//! - **Weakly connected components** — via petgraph's `algo::connected_components` on undirected view
//!
//! All algorithms operate on `CodeGraph` and return results indexed by node ID (String).
//! The Louvain algorithm is implemented from scratch because the `graphina` crate
//! requires Rust 1.86+ (our MSRV target is 1.70+).

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

use super::models::{
    AnalyticsConfig, CodeGraph, CodeHealthReport, CommunityInfo, ComponentInfo, GraphAnalytics,
    NodeMetrics,
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
pub fn louvain_communities(
    graph: &CodeGraph,
    resolution: f64,
) -> (HashMap<String, u32>, Vec<CommunityInfo>, f64) {
    let g = &graph.graph;
    let n = g.node_count();
    if n == 0 {
        return (HashMap::new(), vec![], 0.0);
    }

    // Build undirected adjacency lists (much faster than HashMap<(usize,usize)>)
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut node_strengths: Vec<f64> = vec![0.0; n]; // weighted degree

    for edge in g.edge_references() {
        let s = edge.source().index();
        let t = edge.target().index();
        let w = edge.weight().weight;

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
            });
        }
        return (node_map, communities, 0.0);
    }

    // Initialize: each node in its own community
    let mut community: Vec<u32> = (0..n as u32).collect();

    // Maintain community total strength incrementally
    let mut comm_total_strength: HashMap<u32, f64> = HashMap::with_capacity(n);
    for (i, &ki) in node_strengths.iter().enumerate() {
        *comm_total_strength.entry(community[i]).or_default() += ki;
    }

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = 100;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        for node_idx in 0..n {
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

    // Build CommunityInfo with auto-generated labels
    let mut communities: Vec<CommunityInfo> = comm_members
        .into_iter()
        .map(|(id, members)| {
            let label = generate_community_label(&members);
            CommunityInfo {
                id,
                size: members.len(),
                members,
                label,
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

/// Run all 5 algorithms and assemble a complete `GraphAnalytics` result.
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
    let (comm_map, communities, modularity) = louvain_communities(graph, config.louvain_resolution);

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
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{CodeEdge, CodeEdgeType, CodeNode, CodeNodeType};

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
        let (node_map, communities, modularity) = louvain_communities(&g, 1.0);

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
        let (_, communities, _) = louvain_communities(&g, 1.0);

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
}
