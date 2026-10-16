//! A deterministic, offline conversation graph and three exact structural
//! readings of it. Proof slice of RFC #1225d627 (« le segment comme graphe »).
//!
//! # Construction (integer, deterministic)
//!
//! Nodes are turns; a turn carries a set of labels (its topic/task fingerprints,
//! as the segmenter already observes) and a set of tool names. Two turns are
//! **adjacent** when they are consecutive, or when they share at least one label:
//!
//! ```text
//!   A[i][j] = [ |i-j| == 1 ]  OR  [ labels(i) ∩ labels(j) ≠ ∅ ]
//! ```
//!
//! Labels carry the topology (what the conversation is *about*); tools annotate
//! nodes (what it *did*) but do not create edges, so two unrelated turns that
//! both run `Bash` do not merge. The adjacency is exactly `S + (C·Cᵀ > 0)` where
//! `S` is the consecutive-turn matrix and `C` the turn–label incidence matrix,
//! thresholded back to {0,1}.
//!
//! # The three readings
//!
//! 1. [`ConversationGraph::components`] — connected components of the turn
//!    subgraph: a maximal set of mutually reachable turns. A component is the
//!    *discovered* segment, with no fixed per-axis threshold (π₀ of the graph).
//! 2. [`ConversationGraph::articulation_points`] — turns whose removal increases
//!    the number of components (Tarjan, O(V+E)). A hinge between two subjects.
//! 3. [`ConversationGraph::pagerank`] — the stationary distribution of a random
//!    surfer on the turn subgraph (damping 0.85, power iteration to a fixed
//!    tolerance). Its peak is the turn the conversation keeps coming back to;
//!    [`ConversationGraph::hub`] is the cheaper degree proxy.

#![allow(clippy::needless_range_loop)]
use std::collections::{BTreeMap, BTreeSet};

/// One turn, as the graph builder consumes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnNode {
    pub turn: u32,
    pub labels: BTreeSet<String>,
    pub tools: BTreeSet<String>,
}

/// A connected component of turn nodes: a discovered segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub turns: Vec<u32>,
    pub labels: BTreeSet<String>,
}

/// The conversation as an undirected graph over turns.
#[derive(Debug, Clone)]
pub struct ConversationGraph {
    turns: Vec<u32>,
    labels: BTreeMap<u32, BTreeSet<String>>,
    tools: BTreeMap<u32, BTreeSet<String>>,
    adjacency: BTreeMap<u32, BTreeSet<u32>>,
}

impl ConversationGraph {
    /// Build the graph from turn observations (labels + tools).
    pub fn build(turns: &[TurnNode]) -> Self {
        let mut graph = ConversationGraph {
            turns: Vec::with_capacity(turns.len()),
            labels: BTreeMap::new(),
            tools: BTreeMap::new(),
            adjacency: BTreeMap::new(),
        };
        for t in turns {
            graph.turns.push(t.turn);
            graph.labels.insert(t.turn, t.labels.clone());
            graph.tools.insert(t.turn, t.tools.clone());
            graph.adjacency.insert(t.turn, BTreeSet::new());
        }
        graph.compute_adjacency();
        graph
    }

    /// The turns, in input order.
    pub fn turns(&self) -> &[u32] {
        &self.turns
    }

    /// The labels of a turn, when it is part of the graph.
    pub fn labels_of(&self, turn: u32) -> Option<&BTreeSet<String>> {
        self.labels.get(&turn)
    }

    /// The tools of a turn, when it is part of the graph.
    pub fn tools_of(&self, turn: u32) -> Option<&BTreeSet<String>> {
        self.tools.get(&turn)
    }

    /// The neighbours of a turn (consecutive or sharing a label).
    pub fn neighbours(&self, turn: u32) -> BTreeSet<u32> {
        self.adjacency.get(&turn).cloned().unwrap_or_default()
    }

    fn compute_adjacency(&mut self) {
        let n = self.turns.len();
        for i in 0..n {
            let a = self.turns[i];
            for j in (i + 1)..n {
                let b = self.turns[j];
                let consecutive = b == a + 1;
                let share_label = match (self.labels.get(&a), self.labels.get(&b)) {
                    (Some(la), Some(lb)) => !la.is_disjoint(lb),
                    _ => false,
                };
                if consecutive || share_label {
                    self.adjacency
                        .get_mut(&a)
                        .expect("turn registered")
                        .insert(b);
                    self.adjacency
                        .get_mut(&b)
                        .expect("turn registered")
                        .insert(a);
                }
            }
        }
    }

    /// Connected components of the turn subgraph (BFS, deterministic order).
    pub fn components(&self) -> Vec<Component> {
        let mut visited: BTreeSet<u32> = BTreeSet::new();
        let mut out = Vec::new();
        for &start in &self.turns {
            if visited.contains(&start) {
                continue;
            }
            let mut turns = Vec::new();
            let mut stack = vec![start];
            visited.insert(start);
            while let Some(t) = stack.pop() {
                turns.push(t);
                for n in self.neighbours(t) {
                    if visited.insert(n) {
                        stack.push(n);
                    }
                }
            }
            turns.sort_unstable();
            let labels: BTreeSet<String> = turns
                .iter()
                .filter_map(|t| self.labels.get(t))
                .flatten()
                .cloned()
                .collect();
            out.push(Component { turns, labels });
        }
        out
    }

    /// Articulation points among turn nodes (Tarjan, O(V+E)). A turn is a pivot
    /// when removing it splits a component.
    pub fn articulation_points(&self) -> BTreeSet<u32> {
        let mut visited: BTreeSet<u32> = BTreeSet::new();
        let mut disc: BTreeMap<u32, u32> = BTreeMap::new();
        let mut low: BTreeMap<u32, u32> = BTreeMap::new();
        let mut parent: BTreeMap<u32, Option<u32>> = BTreeMap::new();
        let mut ap: BTreeSet<u32> = BTreeSet::new();
        let mut time = 0u32;

        #[allow(clippy::too_many_arguments)] // Tarjan DFS: six pieces of mutable traversal state are clearer passed individually than bundled
        fn dfs(
            u: u32,
            graph: &ConversationGraph,
            visited: &mut BTreeSet<u32>,
            disc: &mut BTreeMap<u32, u32>,
            low: &mut BTreeMap<u32, u32>,
            parent: &mut BTreeMap<u32, Option<u32>>,
            ap: &mut BTreeSet<u32>,
            time: &mut u32,
        ) {
            visited.insert(u);
            *time += 1;
            disc.insert(u, *time);
            low.insert(u, *time);
            let mut children = 0u32;
            for v in graph.neighbours(u) {
                if !visited.contains(&v) {
                    children += 1;
                    parent.insert(v, Some(u));
                    dfs(v, graph, visited, disc, low, parent, ap, time);
                    let lv = *low.get(&v).expect("low set");
                    let lu = *low.get(&u).expect("low set");
                    low.insert(u, lu.min(lv));
                    let du = *disc.get(&u).expect("disc set");
                    let is_root = parent.get(&u).and_then(|p| *p).is_none();
                    if (is_root && children > 1) || (!is_root && lv >= du) {
                        ap.insert(u);
                    }
                } else if parent.get(&v).and_then(|p| *p) != Some(u) {
                    let dv = *disc.get(&v).expect("disc set");
                    let lu = *low.get(&u).expect("low set");
                    low.insert(u, lu.min(dv));
                }
            }
        }

        for &t in &self.turns {
            if !visited.contains(&t) {
                parent.insert(t, None);
                dfs(
                    t,
                    self,
                    &mut visited,
                    &mut disc,
                    &mut low,
                    &mut parent,
                    &mut ap,
                    &mut time,
                );
            }
        }
        ap
    }

    /// PageRank of the turn subgraph (power iteration, damping, deterministic:
    /// fixed iteration count and tolerance, no randomness).
    pub fn pagerank(&self, damping: f64, tolerance: f64) -> BTreeMap<u32, f64> {
        let n = self.turns.len();
        if n == 0 {
            return BTreeMap::new();
        }
        let mut rank: BTreeMap<u32, f64> =
            self.turns.iter().map(|&t| (t, 1.0 / n as f64)).collect();
        let mut next: BTreeMap<u32, f64> = BTreeMap::new();
        for _ in 0..100 {
            let dangling: f64 = self
                .turns
                .iter()
                .copied()
                .filter(|t| self.adjacency[t].is_empty())
                .map(|t| rank[&t])
                .sum();
            for &t in &self.turns {
                let nb = &self.adjacency[&t];
                let base = if nb.is_empty() { 1.0 / n as f64 } else { 0.0 };
                let mut acc = 0.0;
                for &s in nb {
                    acc += rank[&s] / self.adjacency[&s].len() as f64;
                }
                let val = (1.0 - damping) / n as f64 + damping * (base + acc + dangling / n as f64);
                next.insert(t, val);
            }
            let delta: f64 = self
                .turns
                .iter()
                .map(|&t| (next[&t] - rank[&t]).abs())
                .sum();
            std::mem::swap(&mut rank, &mut next);
            if delta < tolerance {
                break;
            }
        }
        rank
    }

    /// The turn of highest degree (cheap centrality), ties broken to the
    /// smallest turn for determinism.
    pub fn hub(&self) -> Option<u32> {
        self.turns
            .iter()
            .copied()
            .max_by_key(|t| (self.adjacency[t].len(), std::cmp::Reverse(*t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn t(turn: u32, labels: &[&str], tools: &[&str]) -> TurnNode {
        TurnNode {
            turn,
            labels: labels.iter().map(|s| s.to_string()).collect(),
            tools: tools.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn a_bridge_turn_is_the_single_articulation_point_and_hub() {
        // Two label-cliques ("routing" 0..=2, "auth" 4..=6) joined by turn 3,
        // which carries both labels: it is the sole hinge and the densest node.
        let turns = vec![
            t(0, &["routing"], &["Read"]),
            t(1, &["routing"], &["Bash"]),
            t(2, &["routing"], &["Read"]),
            t(3, &["routing", "auth"], &["Edit"]),
            t(4, &["auth"], &["Edit"]),
            t(5, &["auth"], &["Bash"]),
            t(6, &["auth"], &["Read"]),
        ];
        let g = ConversationGraph::build(&turns);
        assert_eq!(
            g.components().len(),
            1,
            "one theme glued by the bridge turn"
        );
        assert_eq!(g.articulation_points(), BTreeSet::from([3]));
        assert_eq!(g.hub(), Some(3));
        let pr = g.pagerank(0.85, 1e-9);
        let peak = pr.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(k, _)| *k);
        assert_eq!(peak, Some(3), "the bridge turn is the stationary peak");
    }

    #[test]
    fn two_disjoint_themes_form_two_components() {
        // "routing" (0,1) and "auth" (5,6) with a gap: no consecutive edge and
        // no shared label, so two components and no hinge.
        let turns = vec![
            t(0, &["routing"], &["Read"]),
            t(1, &["routing"], &["Bash"]),
            t(5, &["auth"], &["Edit"]),
            t(6, &["auth"], &["Bash"]),
        ];
        let g = ConversationGraph::build(&turns);
        assert_eq!(g.components().len(), 2);
        assert!(g.articulation_points().is_empty());
    }

    #[test]
    fn tools_annotate_but_do_not_glue() {
        // Two turns that share a tool but no label and are not consecutive stay
        // in different components: behaviour does not forge the topology.
        let turns = vec![t(0, &["routing"], &["Bash"]), t(3, &["auth"], &["Bash"])];
        let g = ConversationGraph::build(&turns);
        assert_eq!(g.components().len(), 2);
    }

    #[test]
    fn the_graph_is_deterministic() {
        let mk = || {
            vec![
                t(0, &["routing"], &["Read"]),
                t(1, &["routing", "auth"], &["Bash"]),
                t(2, &["auth"], &["Edit"]),
            ]
        };
        let a = ConversationGraph::build(&mk());
        let b = ConversationGraph::build(&mk());
        assert_eq!(a.components(), b.components());
        assert_eq!(a.articulation_points(), b.articulation_points());
        assert_eq!(a.pagerank(0.85, 1e-9), b.pagerank(0.85, 1e-9));
    }
}

/// The spectral cut of the conversation graph: the Fiedler vector (eigenvector
/// of the second-smallest Laplacian eigenvalue λ₂, the algebraic connectivity)
/// and the two sides it partitions the turns into.
#[derive(Debug, Clone, PartialEq)]
pub struct FiedlerCut {
    pub eigenvalue: f64,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
}

impl ConversationGraph {
    /// The graph Laplacian `L = D − A` as a dense symmetric matrix, in turn
    /// order (`self.turns()`).
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.turns.len();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            l[i][i] = self.adjacency[&self.turns[i]].len() as f64;
        }
        for i in 0..n {
            let ti = self.turns[i];
            for j in (i + 1)..n {
                if self.adjacency[&ti].contains(&self.turns[j]) {
                    l[i][j] = -1.0;
                    l[j][i] = -1.0;
                }
            }
        }
        l
    }

    /// The Fiedler cut: the eigenvector of the second-smallest Laplacian
    /// eigenvalue, by the Jacobi method, split by sign. Returns `None` when the
    /// graph is already disconnected (λ₂ ≈ 0: no single cut to make) or has
    /// fewer than two turns.
    pub fn fiedler_cut(&self, tolerance: f64) -> Option<FiedlerCut> {
        let n = self.turns.len();
        if n < 2 {
            return None;
        }
        let (values, vectors) = jacobi_eigen(self.laplacian(), tolerance, 100);
        let lambda2 = *values.get(1)?;
        if lambda2.abs() < tolerance {
            return None;
        }
        let fiedler = &vectors[1];
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, &x) in fiedler.iter().enumerate() {
            if x >= 0.0 {
                left.push(self.turns[i]);
            } else {
                right.push(self.turns[i]);
            }
        }
        // Canonical: the side holding the smallest turn is `left`.
        let left_min = left.first().copied().unwrap_or(u32::MAX);
        let right_min = right.first().copied().unwrap_or(u32::MAX);
        if left_min > right_min {
            std::mem::swap(&mut left, &mut right);
        }
        Some(FiedlerCut {
            eigenvalue: lambda2,
            left,
            right,
        })
    }
}

/// Jacobi eigenvalue algorithm for a real symmetric matrix: eigenvalues sorted
/// ascending and the matching eigenvectors (columns). Deterministic — fixed
/// rotation order, fixed sweep and tolerance budget, no randomness.
fn jacobi_eigen(
    mut a: Vec<Vec<f64>>,
    tolerance: f64,
    max_sweeps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off.sqrt() < tolerance {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < tolerance {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut eigen: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| (a[i][i], (0..n).map(|k| v[k][i]).collect()))
        .collect();
    eigen.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let values = eigen.iter().map(|e| e.0).collect();
    let vectors = eigen.into_iter().map(|e| e.1).collect();
    (values, vectors)
}

#[cfg(test)]
mod spectral_tests {
    use super::*;

    fn t(turn: u32, labels: &[&str], tools: &[&str]) -> TurnNode {
        TurnNode {
            turn,
            labels: labels.iter().map(|s| s.to_string()).collect(),
            tools: tools.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn the_path_graph_p3_has_fiedler_value_one() {
        // Turns 0-1-2 consecutive, no shared label: the path P3, whose
        // Laplacian spectrum is {0, 1, 3}, so λ₂ = 1.
        let turns = vec![t(0, &["a"], &[]), t(1, &["b"], &[]), t(2, &["c"], &[])];
        let g = ConversationGraph::build(&turns);
        let cut = g.fiedler_cut(1e-9).expect("connected graph has a cut");
        assert!(
            (cut.eigenvalue - 1.0).abs() < 1e-6,
            "λ₂(P3) = 1, got {}",
            cut.eigenvalue
        );
    }

    #[test]
    fn the_fiedler_cut_separates_the_two_themes() {
        // Clique "routing" (0,1,2) and clique "auth" (4,5,6) joined by the
        // single path 2—3 (turn 3 shares "bridge" with 2 and "auth" with 4..6).
        let turns = vec![
            t(0, &["routing"], &[]),
            t(1, &["routing"], &[]),
            t(2, &["routing", "bridge"], &[]),
            t(3, &["bridge", "auth"], &[]),
            t(4, &["auth"], &[]),
            t(5, &["auth"], &[]),
            t(6, &["auth"], &[]),
        ];
        let g = ConversationGraph::build(&turns);
        let cut = g.fiedler_cut(1e-9).expect("connected graph has a cut");
        assert!(cut.eigenvalue > 0.0, "λ₂ must be positive");
        let routing_side = cut.left.contains(&0);
        for &turn in &[1, 2] {
            assert_eq!(
                cut.left.contains(&turn),
                routing_side,
                "turn {turn} must stay with turn 0"
            );
        }
        for &turn in &[4, 5, 6] {
            assert_eq!(
                cut.left.contains(&turn),
                !routing_side,
                "turn {turn} must sit opposite turn 0"
            );
        }
    }

    #[test]
    fn a_disconnected_graph_has_no_fiedler_cut() {
        let turns = vec![
            t(0, &["routing"], &[]),
            t(1, &["routing"], &[]),
            t(5, &["auth"], &[]),
            t(6, &["auth"], &[]),
        ];
        let g = ConversationGraph::build(&turns);
        assert!(g.fiedler_cut(1e-9).is_none());
    }
}

impl ConversationGraph {
    /// Newman–Girvan modularity of a partition (undirected):
    /// `Q = Σ_c [ L_c/m − (K_c/(2m))² ]`, where `L_c` is the number of edges
    /// inside community `c`, `K_c` the sum of its members' degrees, and `m` the
    /// total number of edges. A graph with no edge scores 0.
    pub fn modularity(&self, communities: &[Vec<u32>]) -> f64 {
        let m: f64 = self
            .turns
            .iter()
            .map(|t| self.adjacency[t].len())
            .sum::<usize>() as f64
            / 2.0;
        if m == 0.0 {
            return 0.0;
        }
        let mut q = 0.0;
        for community in communities {
            let members: BTreeSet<u32> = community.iter().copied().collect();
            let mut internal = 0usize;
            let mut degree_sum = 0usize;
            for &t in community {
                degree_sum += self.adjacency[&t].len();
                internal += self.adjacency[&t]
                    .iter()
                    .filter(|nb| members.contains(*nb))
                    .count();
            }
            let l_c = internal as f64 / 2.0;
            let k_c = degree_sum as f64;
            q += l_c / m - (k_c / (2.0 * m)).powi(2);
        }
        q
    }

    /// Split `nodes` by the sign of the Fiedler vector of their induced
    /// subgraph. `None` when the subgraph is already split (λ₂ ≈ 0) or has
    /// fewer than two nodes.
    fn bisect(&self, nodes: &[u32], tolerance: f64) -> Option<(Vec<u32>, Vec<u32>)> {
        if nodes.len() < 2 {
            return None;
        }
        let n = nodes.len();
        let index: BTreeMap<u32, usize> = nodes.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let mut l = vec![vec![0.0; n]; n];
        for (i, &t) in nodes.iter().enumerate() {
            let degree = self.adjacency[&t]
                .iter()
                .filter(|nb| index.contains_key(*nb))
                .count();
            l[i][i] = degree as f64;
            for nb in &self.adjacency[&t] {
                if let Some(&j) = index.get(nb) {
                    l[i][j] = -1.0;
                }
            }
        }
        let (values, vectors) = jacobi_eigen(l, tolerance, 100);
        let lambda2 = *values.get(1)?;
        if lambda2.abs() < tolerance {
            return None;
        }
        let fiedler = &vectors[1];
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, &x) in fiedler.iter().enumerate() {
            if x >= 0.0 {
                left.push(nodes[i]);
            } else {
                right.push(nodes[i]);
            }
        }
        if left.is_empty() || right.is_empty() {
            return None;
        }
        Some((left, right))
    }

    /// Recursive spectral bisection with a modularity stopping rule: a
    /// community is split further only while it strictly improves `Q` by more
    /// than `improvement`. Deterministic; one turn list per segment.
    pub fn spectral_segments(&self, improvement: f64) -> Vec<Vec<u32>> {
        if self.turns.is_empty() {
            return Vec::new();
        }
        let mut communities: Vec<Vec<u32>> = vec![self.turns.clone()];
        loop {
            let base_q = self.modularity(&communities);
            let mut best: Option<(f64, Vec<Vec<u32>>)> = None;
            for (idx, community) in communities.iter().enumerate() {
                if let Some((left, right)) = self.bisect(community, 1e-9) {
                    let mut candidate = communities.clone();
                    candidate.splice(idx..=idx, [left, right]);
                    let q = self.modularity(&candidate);
                    let improves = q > base_q + improvement;
                    let is_best = best.as_ref().is_none_or(|(best_q, _)| q > *best_q);
                    if improves && is_best {
                        best = Some((q, candidate));
                    }
                }
            }
            match best {
                Some((_, next)) => communities = next,
                None => break,
            }
        }
        communities.sort_by_key(|c| c.first().copied().unwrap_or(u32::MAX));
        communities
    }
}

#[cfg(test)]
mod partition_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn t(turn: u32, labels: &[&str]) -> TurnNode {
        TurnNode {
            turn,
            labels: labels.iter().map(|s| s.to_string()).collect(),
            tools: BTreeSet::new(),
        }
    }

    /// Two label-cliques (0,1,2 routing / 3,4,5 auth) joined by the single
    /// consecutive edge 2—3: a barbell with 7 edges.
    fn barbell() -> ConversationGraph {
        ConversationGraph::build(&[
            t(0, &["routing"]),
            t(1, &["routing"]),
            t(2, &["routing"]),
            t(3, &["auth"]),
            t(4, &["auth"]),
            t(5, &["auth"]),
        ])
    }

    #[test]
    fn modularity_rewards_the_two_themes_and_zeros_the_single_block() {
        let g = barbell();
        let split = g.modularity(&[vec![0, 1, 2], vec![3, 4, 5]]);
        let whole = g.modularity(&[vec![0, 1, 2, 3, 4, 5]]);
        assert!(split > 0.3, "the barbell split scores ~0.357, got {split}");
        assert!(
            whole.abs() < 1e-9,
            "one community always scores 0, got {whole}"
        );
    }

    #[test]
    fn spectral_segments_finds_the_two_themes() {
        let g = barbell();
        let segments = g.spectral_segments(1e-9);
        assert_eq!(segments.len(), 2, "two themes, two segments: {segments:?}");
        assert_eq!(segments[0], vec![0, 1, 2]);
        assert_eq!(segments[1], vec![3, 4, 5]);
    }

    #[test]
    fn spectral_segments_is_deterministic() {
        let a = barbell().spectral_segments(1e-9);
        let b = barbell().spectral_segments(1e-9);
        assert_eq!(a, b);
    }
}
