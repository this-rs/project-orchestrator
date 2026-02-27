//! Process detection — entry point scoring, BFS traces, and framework awareness.
//!
//! Discovers business processes by:
//! 1. Scoring functions as potential entry points (multi-criteria)
//! 2. Detecting web framework patterns from file paths
//! 3. BFS traversal from entry points through the CALLS graph
//! 4. Deduplication (subset removal + endpoint dedup)
//! 5. Classification (intra vs cross community)

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::visit::EdgeRef;

use crate::graph::models::{CodeEdgeType, CodeGraph, CodeNodeType, NodeMetrics};

// ── Entry Point Scoring ─────────────────────────────────────────────

/// Score components for an entry point candidate.
#[derive(Debug, Clone)]
pub struct EntryPointScore {
    /// Final composite score.
    pub score: f64,
    /// Human-readable reason for the score.
    pub reason: String,
    /// base = out_degree / (in_degree + 1)
    pub base_score: f64,
    /// 2.0 if exported (public), 1.0 otherwise
    pub export_multiplier: f64,
    /// Pattern-based name multiplier (0.3–3.0)
    pub name_multiplier: f64,
    /// Framework-based multiplier (1.0–3.0)
    pub framework_multiplier: f64,
}

/// Name patterns that indicate entry-point functions (high multiplier).
const ENTRY_POINT_NAMES: &[&str] = &[
    "main",
    "run",
    "start",
    "init",
    "execute",
    "launch",
    "boot",
    "setup",
    "serve",
    "listen",
    "handle",
    "dispatch",
    "process",
    "route",
    "handler",
    "controller",
    "endpoint",
    "middleware",
    "cli",
    "command",
    "cmd",
    "test_",
    "spec_",
    "bench_",
];

/// Name patterns that indicate utility/helper functions (penalty).
const UTILITY_NAMES: &[&str] = &[
    "get",
    "set",
    "is_",
    "has_",
    "can_",
    "should_",
    "to_",
    "from_",
    "into_",
    "as_",
    "format",
    "parse",
    "validate",
    "sanitize",
    "normalize",
    "log",
    "debug",
    "trace",
    "warn",
    "error",
    "info",
    "helper",
    "util",
    "utils",
    "internal",
    "private",
    "new",
    "default",
    "clone",
    "drop",
    "fmt",
    "eq",
    "serialize",
    "deserialize",
];

/// Score all functions in the graph as entry point candidates.
///
/// Returns a map from node ID to `EntryPointScore`.
pub fn score_entry_points(
    graph: &CodeGraph,
    metrics: &HashMap<String, NodeMetrics>,
) -> HashMap<String, EntryPointScore> {
    let mut scores = HashMap::new();

    for (id, node_metrics) in metrics {
        // Only score Function nodes
        let node = match graph.get_node(id) {
            Some(n) if n.node_type == CodeNodeType::Function => n,
            _ => continue,
        };

        let name = &node.name;
        let name_lower = name.to_lowercase();

        // Base score: out_degree / (in_degree + 1)
        // Functions that call many but are called by few → high score
        let base_score = node_metrics.out_degree as f64 / (node_metrics.in_degree as f64 + 1.0);

        // Export multiplier: public functions are more likely entry points
        // We use in_degree == 0 as a proxy for "not called internally" combined
        // with the visibility check from the node name conventions.
        // Default to 1.0; boost if the function has no internal callers.
        let export_multiplier = if node_metrics.in_degree == 0 {
            2.0
        } else {
            1.0
        };

        // Name multiplier
        let name_multiplier = compute_name_multiplier(&name_lower);

        // Framework multiplier (from file path)
        let framework_multiplier = node
            .path
            .as_deref()
            .and_then(detect_framework)
            .map(|(mult, _)| mult)
            .unwrap_or(1.0);

        let score = base_score * export_multiplier * name_multiplier * framework_multiplier;

        let mut reasons = Vec::new();
        if base_score > 1.0 {
            reasons.push(format!("high fan-out ratio ({:.1})", base_score));
        }
        if export_multiplier > 1.0 {
            reasons.push("no internal callers".to_string());
        }
        if name_multiplier > 1.5 {
            reasons.push(format!(
                "entry-point name pattern (×{:.1})",
                name_multiplier
            ));
        } else if name_multiplier < 0.5 {
            reasons.push(format!("utility name penalty (×{:.1})", name_multiplier));
        }
        if framework_multiplier > 1.0 {
            let fw_name = node
                .path
                .as_deref()
                .and_then(detect_framework)
                .map(|(_, name)| name)
                .unwrap_or("unknown");
            reasons.push(format!(
                "{} framework (×{:.1})",
                fw_name, framework_multiplier
            ));
        }

        let reason = if reasons.is_empty() {
            "default scoring".to_string()
        } else {
            reasons.join(", ")
        };

        scores.insert(
            id.clone(),
            EntryPointScore {
                score,
                reason,
                base_score,
                export_multiplier,
                name_multiplier,
                framework_multiplier,
            },
        );
    }

    scores
}

/// Compute the name-based multiplier for a function name.
fn compute_name_multiplier(name_lower: &str) -> f64 {
    // Check entry-point patterns first (higher priority)
    for pattern in ENTRY_POINT_NAMES {
        if name_lower.starts_with(pattern) || name_lower.ends_with(pattern) {
            return 3.0;
        }
    }

    // Check utility patterns (penalty)
    for pattern in UTILITY_NAMES {
        if name_lower.starts_with(pattern) || name_lower.ends_with(pattern) {
            return 0.3;
        }
    }

    // Default: neutral
    1.0
}

// ── Framework Detection ─────────────────────────────────────────────

/// Framework detection result: (multiplier, framework name).
struct FrameworkPattern {
    /// Substring to match in the file path.
    path_contains: &'static str,
    /// Score multiplier for entry points in this framework.
    multiplier: f64,
    /// Framework name for display.
    name: &'static str,
}

/// All known framework patterns, ordered by specificity (most specific first).
const FRAMEWORK_PATTERNS: &[FrameworkPattern] = &[
    // Next.js (pages/api is highest specificity)
    FrameworkPattern {
        path_contains: "pages/api/",
        multiplier: 3.0,
        name: "Next.js API",
    },
    FrameworkPattern {
        path_contains: "app/api/",
        multiplier: 3.0,
        name: "Next.js App Router",
    },
    FrameworkPattern {
        path_contains: "pages/",
        multiplier: 1.5,
        name: "Next.js Pages",
    },
    // Express / Fastify / Node.js
    FrameworkPattern {
        path_contains: "/routes/",
        multiplier: 2.5,
        name: "Express/Node",
    },
    FrameworkPattern {
        path_contains: "/middleware/",
        multiplier: 2.0,
        name: "Express Middleware",
    },
    // Django
    FrameworkPattern {
        path_contains: "views.py",
        multiplier: 3.0,
        name: "Django",
    },
    FrameworkPattern {
        path_contains: "urls.py",
        multiplier: 2.5,
        name: "Django URLs",
    },
    // Flask
    FrameworkPattern {
        path_contains: "/blueprints/",
        multiplier: 2.5,
        name: "Flask",
    },
    // Spring Boot / Java
    FrameworkPattern {
        path_contains: "/controllers/",
        multiplier: 3.0,
        name: "Spring/MVC",
    },
    FrameworkPattern {
        path_contains: "/controller/",
        multiplier: 3.0,
        name: "Spring/MVC",
    },
    FrameworkPattern {
        path_contains: "/rest/",
        multiplier: 2.5,
        name: "REST API",
    },
    // Laravel
    FrameworkPattern {
        path_contains: "Controllers/",
        multiplier: 3.0,
        name: "Laravel",
    },
    // Rails
    FrameworkPattern {
        path_contains: "app/controllers/",
        multiplier: 2.5,
        name: "Rails",
    },
    // Rust web (Actix/Axum)
    FrameworkPattern {
        path_contains: "/handlers/",
        multiplier: 2.5,
        name: "Rust Web",
    },
    FrameworkPattern {
        path_contains: "/handlers.rs",
        multiplier: 2.5,
        name: "Rust Web",
    },
    FrameworkPattern {
        path_contains: "/api/",
        multiplier: 2.0,
        name: "API",
    },
    // Go
    FrameworkPattern {
        path_contains: "handler.go",
        multiplier: 2.5,
        name: "Go HTTP",
    },
    FrameworkPattern {
        path_contains: "handlers.go",
        multiplier: 2.5,
        name: "Go HTTP",
    },
    // Frontend frameworks
    FrameworkPattern {
        path_contains: "/components/",
        multiplier: 1.5,
        name: "Component",
    },
    FrameworkPattern {
        path_contains: "/views/",
        multiplier: 1.5,
        name: "View",
    },
    FrameworkPattern {
        path_contains: ".component.ts",
        multiplier: 1.5,
        name: "Angular",
    },
    FrameworkPattern {
        path_contains: ".service.ts",
        multiplier: 1.5,
        name: "Angular Service",
    },
    // CLI
    FrameworkPattern {
        path_contains: "/commands/",
        multiplier: 2.5,
        name: "CLI",
    },
    FrameworkPattern {
        path_contains: "/cmd/",
        multiplier: 2.5,
        name: "CLI",
    },
    FrameworkPattern {
        path_contains: "cli.rs",
        multiplier: 2.5,
        name: "CLI",
    },
    FrameworkPattern {
        path_contains: "cli.py",
        multiplier: 2.5,
        name: "CLI",
    },
    FrameworkPattern {
        path_contains: "cli.ts",
        multiplier: 2.5,
        name: "CLI",
    },
];

/// Detect the framework from a file path.
///
/// Returns `(multiplier, framework_name)` if a pattern matches, `None` otherwise.
pub fn detect_framework(file_path: &str) -> Option<(f64, &'static str)> {
    for pattern in FRAMEWORK_PATTERNS {
        if file_path.contains(pattern.path_contains) {
            return Some((pattern.multiplier, pattern.name));
        }
    }
    None
}

// ── BFS Trace Engine ────────────────────────────────────────────────

/// Configuration for process detection BFS.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Maximum depth for BFS traversal.
    pub max_trace_depth: usize,
    /// Maximum branching factor (top N callees by PageRank).
    pub max_branching: usize,
    /// Maximum number of processes to detect.
    pub max_processes: usize,
    /// Minimum steps for a valid trace.
    pub min_steps: usize,
    /// Minimum edge confidence (weight) to follow.
    pub min_trace_confidence: f64,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            max_trace_depth: 10,
            max_branching: 4,
            max_processes: 75,
            min_steps: 3,
            min_trace_confidence: 0.5,
        }
    }
}

/// A single process trace discovered by BFS.
#[derive(Debug, Clone)]
pub struct ProcessTrace {
    /// ID of the entry point function.
    pub entry_point_id: String,
    /// ID of the terminal function.
    pub terminal_id: String,
    /// Ordered list of function IDs in this trace.
    pub steps: Vec<String>,
    /// Community IDs traversed by this trace.
    pub community_ids: HashSet<u32>,
}

/// Classification of a detected process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessType {
    /// All steps are in the same community.
    IntraCommunity,
    /// Steps span multiple communities.
    CrossCommunity,
}

impl std::fmt::Display for ProcessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessType::IntraCommunity => write!(f, "intra_community"),
            ProcessType::CrossCommunity => write!(f, "cross_community"),
        }
    }
}

/// A fully classified process ready for persistence.
#[derive(Debug, Clone)]
pub struct Process {
    /// Unique identifier.
    pub id: String,
    /// Auto-generated label from entry point name.
    pub label: String,
    /// Classification: intra or cross community.
    pub process_type: ProcessType,
    /// Ordered list of function IDs.
    pub steps: Vec<String>,
    /// Entry point function ID.
    pub entry_point_id: String,
    /// Terminal function ID.
    pub terminal_id: String,
    /// Community IDs traversed.
    pub communities: HashSet<u32>,
}

/// Run BFS from a single entry point, collecting traces.
fn bfs_trace_single(
    graph: &CodeGraph,
    start: &str,
    config: &ProcessConfig,
    metrics: &HashMap<String, NodeMetrics>,
) -> Vec<ProcessTrace> {
    let start_idx = match graph.get_index(start) {
        Some(idx) => idx,
        None => return Vec::new(),
    };

    let mut traces = Vec::new();

    // BFS with path tracking: (current_index, path_so_far)
    let mut queue: VecDeque<(petgraph::graph::NodeIndex, Vec<String>)> = VecDeque::new();
    queue.push_back((start_idx, vec![start.to_string()]));

    while let Some((current_idx, path)) = queue.pop_front() {
        if path.len() > config.max_trace_depth {
            // Terminal by depth limit — record trace
            let trace = build_trace(&path, metrics);
            traces.push(trace);
            continue;
        }

        // Get outgoing CALLS edges
        let mut callees: Vec<(petgraph::graph::NodeIndex, &str, f64)> = Vec::new();
        for edge_ref in graph
            .graph
            .edges_directed(current_idx, petgraph::Direction::Outgoing)
        {
            let edge = edge_ref.weight();
            if edge.edge_type != CodeEdgeType::Calls {
                continue;
            }
            if edge.weight < config.min_trace_confidence {
                continue;
            }
            let target_idx = edge_ref.target();
            let target_node = &graph.graph[target_idx];

            // Skip if already in path (cycle avoidance)
            if path.contains(&target_node.id) {
                continue;
            }

            callees.push((target_idx, &target_node.id, edge.weight));
        }

        if callees.is_empty() {
            // Terminal node — record trace if long enough
            if path.len() >= config.min_steps {
                let trace = build_trace(&path, metrics);
                traces.push(trace);
            }
            continue;
        }

        // Select top N callees by PageRank
        callees.sort_by(|a, b| {
            let pr_a = metrics.get(a.1).map(|m| m.pagerank).unwrap_or(0.0);
            let pr_b = metrics.get(b.1).map(|m| m.pagerank).unwrap_or(0.0);
            pr_b.partial_cmp(&pr_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        callees.truncate(config.max_branching);

        for (target_idx, _, _) in &callees {
            let target_node = &graph.graph[*target_idx];
            let mut new_path = path.clone();
            new_path.push(target_node.id.clone());
            queue.push_back((*target_idx, new_path));
        }
    }

    traces
}

/// Build a ProcessTrace from a path of node IDs.
fn build_trace(path: &[String], metrics: &HashMap<String, NodeMetrics>) -> ProcessTrace {
    let community_ids: HashSet<u32> = path
        .iter()
        .filter_map(|id| metrics.get(id).map(|m| m.community_id))
        .collect();

    ProcessTrace {
        entry_point_id: path.first().cloned().unwrap_or_default(),
        terminal_id: path.last().cloned().unwrap_or_default(),
        steps: path.to_vec(),
        community_ids,
    }
}

/// Run BFS from all entry points, sorted by score descending.
pub fn bfs_trace_all(
    graph: &CodeGraph,
    entry_points: &HashMap<String, EntryPointScore>,
    config: &ProcessConfig,
    metrics: &HashMap<String, NodeMetrics>,
) -> Vec<ProcessTrace> {
    // Sort entry points by score descending
    let mut sorted: Vec<(&String, &EntryPointScore)> = entry_points.iter().collect();
    sorted.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut all_traces = Vec::new();

    for (id, _) in sorted {
        if all_traces.len() >= config.max_processes {
            break;
        }

        let traces = bfs_trace_single(graph, id, config, metrics);
        for trace in traces {
            if trace.steps.len() >= config.min_steps {
                all_traces.push(trace);
            }
            if all_traces.len() >= config.max_processes {
                break;
            }
        }
    }

    all_traces
}

// ── Deduplication ───────────────────────────────────────────────────

/// Pass 1: Remove traces that are strict subsets of longer traces.
pub fn deduplicate_subset(traces: &mut Vec<ProcessTrace>) {
    // Sort by length descending
    traces.sort_by(|a, b| b.steps.len().cmp(&a.steps.len()));

    let mut retained: Vec<ProcessTrace> = Vec::new();

    for trace in traces.drain(..) {
        let steps_set: HashSet<&String> = trace.steps.iter().collect();
        let is_subset = retained.iter().any(|kept| {
            let kept_set: HashSet<&String> = kept.steps.iter().collect();
            steps_set.is_subset(&kept_set)
        });

        if !is_subset {
            retained.push(trace);
        }
    }

    *traces = retained;
}

/// Pass 2: Keep only the longest trace per (entry, terminal) pair.
pub fn deduplicate_endpoints(traces: &mut Vec<ProcessTrace>) {
    let mut best: HashMap<(String, String), ProcessTrace> = HashMap::new();

    for trace in traces.drain(..) {
        let key = (trace.entry_point_id.clone(), trace.terminal_id.clone());
        match best.get(&key) {
            Some(existing) if existing.steps.len() >= trace.steps.len() => {
                // Keep existing (longer or equal)
            }
            _ => {
                best.insert(key, trace);
            }
        }
    }

    *traces = best.into_values().collect();
}

// ── Classification ──────────────────────────────────────────────────

/// Classify traces into Process objects with intra/cross community labels.
pub fn classify_processes(traces: Vec<ProcessTrace>, graph: &CodeGraph) -> Vec<Process> {
    traces
        .into_iter()
        .enumerate()
        .map(|(i, trace)| {
            let process_type = if trace.community_ids.len() <= 1 {
                ProcessType::IntraCommunity
            } else {
                ProcessType::CrossCommunity
            };

            // Generate label from entry point name
            let label = graph
                .get_node(&trace.entry_point_id)
                .map(|n| format!("Process: {}", n.name))
                .unwrap_or_else(|| format!("Process #{}", i + 1));

            let id = format!(
                "process-{}-{}",
                trace.entry_point_id.replace(['/', ':', '.'], "-"),
                i
            );

            Process {
                id,
                label,
                process_type,
                steps: trace.steps,
                entry_point_id: trace.entry_point_id,
                terminal_id: trace.terminal_id,
                communities: trace.community_ids,
            }
        })
        .collect()
}

// ── Full Pipeline ───────────────────────────────────────────────────

/// Run the complete process detection pipeline.
///
/// 1. Score entry points
/// 2. BFS trace from top entry points
/// 3. Deduplicate (subset + endpoint)
/// 4. Classify (intra vs cross community)
pub fn detect_processes(
    graph: &CodeGraph,
    metrics: &HashMap<String, NodeMetrics>,
    config: &ProcessConfig,
) -> Vec<Process> {
    let entry_points = score_entry_points(graph, metrics);
    let mut traces = bfs_trace_all(graph, &entry_points, config, metrics);

    deduplicate_subset(&mut traces);
    deduplicate_endpoints(&mut traces);

    classify_processes(traces, graph)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{CodeEdge, CodeNode};

    /// Helper to create a Function node.
    fn func_node(id: &str, name: &str, path: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            node_type: CodeNodeType::Function,
            path: Some(path.to_string()),
            name: name.to_string(),
            project_id: None,
        }
    }

    /// Helper to create a CALLS edge.
    fn calls_edge(weight: f64) -> CodeEdge {
        CodeEdge {
            edge_type: CodeEdgeType::Calls,
            weight,
        }
    }

    /// Helper to create default NodeMetrics with given in/out degree.
    fn node_metrics(in_deg: usize, out_deg: usize, community: u32) -> NodeMetrics {
        NodeMetrics {
            pagerank: 0.1,
            betweenness: 0.0,
            community_id: community,
            clustering_coefficient: 0.0,
            component_id: 0,
            in_degree: in_deg,
            out_degree: out_deg,
        }
    }

    // ── Entry Point Scoring Tests ───────────────────────────────

    #[test]
    fn test_entry_point_star_graph() {
        // Star graph: center → leaf1, leaf2, leaf3
        // Center has high out_degree, low in_degree → high base score
        // Leaves have high in_degree, low out_degree → low base score
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("center", "dispatch", "src/main.rs"));
        graph.add_node(func_node("leaf1", "handle_a", "src/handlers.rs"));
        graph.add_node(func_node("leaf2", "handle_b", "src/handlers.rs"));
        graph.add_node(func_node("leaf3", "handle_c", "src/handlers.rs"));
        graph.add_edge("center", "leaf1", calls_edge(1.0));
        graph.add_edge("center", "leaf2", calls_edge(1.0));
        graph.add_edge("center", "leaf3", calls_edge(1.0));

        let mut metrics = HashMap::new();
        metrics.insert("center".into(), node_metrics(0, 3, 0));
        metrics.insert("leaf1".into(), node_metrics(1, 0, 0));
        metrics.insert("leaf2".into(), node_metrics(1, 0, 0));
        metrics.insert("leaf3".into(), node_metrics(1, 0, 0));

        let scores = score_entry_points(&graph, &metrics);

        let center_score = scores.get("center").unwrap().score;
        let leaf_score = scores.get("leaf1").unwrap().score;

        // Center: base=3/1=3.0, export=2.0 (in_deg=0), name=3.0 (dispatch→handler pattern)
        // Leaf: base=0/2=0.0
        assert!(
            center_score > leaf_score,
            "center ({}) should score higher than leaf ({})",
            center_score,
            leaf_score
        );
    }

    #[test]
    fn test_entry_point_name_multiplier() {
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("f_main", "main", "src/main.rs"));
        graph.add_node(func_node("f_handle", "handleRequest", "src/server.rs"));
        graph.add_node(func_node("f_get", "getUserById", "src/utils.rs"));
        graph.add_node(func_node("f_is", "is_valid", "src/utils.rs"));
        graph.add_node(func_node("f_neutral", "transform_data", "src/core.rs"));

        let mut metrics = HashMap::new();
        for id in ["f_main", "f_handle", "f_get", "f_is", "f_neutral"] {
            metrics.insert(id.into(), node_metrics(1, 1, 0));
        }

        let scores = score_entry_points(&graph, &metrics);

        // main and handle → high multiplier (3.0)
        assert!(scores["f_main"].name_multiplier > 1.5);
        assert!(scores["f_handle"].name_multiplier > 1.5);

        // get/is → penalty (0.3)
        assert!(scores["f_get"].name_multiplier < 0.5);
        assert!(scores["f_is"].name_multiplier < 0.5);

        // neutral → default (1.0)
        assert!((scores["f_neutral"].name_multiplier - 1.0).abs() < 0.01);
    }

    // ── Framework Detection Tests ───────────────────────────────

    #[test]
    fn test_detect_framework_nextjs() {
        let (mult, name) = detect_framework("pages/api/users.ts").unwrap();
        assert!((mult - 3.0).abs() < 0.01);
        assert_eq!(name, "Next.js API");
    }

    #[test]
    fn test_detect_framework_spring() {
        let (mult, name) = detect_framework("src/controllers/UserController.java").unwrap();
        assert!((mult - 3.0).abs() < 0.01);
        assert!(name.contains("Spring") || name.contains("MVC"));
    }

    #[test]
    fn test_detect_framework_express() {
        let (mult, _) = detect_framework("src/routes/auth.js").unwrap();
        assert!((mult - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_detect_framework_django() {
        let (mult, name) = detect_framework("myapp/views.py").unwrap();
        assert!((mult - 3.0).abs() < 0.01);
        assert_eq!(name, "Django");
    }

    #[test]
    fn test_detect_framework_none() {
        let result = detect_framework("src/utils/helpers.ts");
        assert!(result.is_none());
    }

    // ── BFS Trace Tests ─────────────────────────────────────────

    #[test]
    fn test_bfs_chain() {
        // Chain: A → B → C → D (4 steps)
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("A", "main", "src/main.rs"));
        graph.add_node(func_node("B", "process", "src/core.rs"));
        graph.add_node(func_node("C", "validate", "src/core.rs"));
        graph.add_node(func_node("D", "save", "src/db.rs"));
        graph.add_edge("A", "B", calls_edge(1.0));
        graph.add_edge("B", "C", calls_edge(1.0));
        graph.add_edge("C", "D", calls_edge(1.0));

        let mut metrics = HashMap::new();
        metrics.insert("A".into(), node_metrics(0, 1, 0));
        metrics.insert("B".into(), node_metrics(1, 1, 0));
        metrics.insert("C".into(), node_metrics(1, 1, 0));
        metrics.insert("D".into(), node_metrics(1, 0, 0));

        let config = ProcessConfig {
            min_steps: 2,
            ..Default::default()
        };

        let traces = bfs_trace_single(&graph, "A", &config, &metrics);
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].steps, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn test_bfs_cycle_avoidance() {
        // Cycle: A → B → C → A
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("A", "start", "src/main.rs"));
        graph.add_node(func_node("B", "process", "src/core.rs"));
        graph.add_node(func_node("C", "loop_back", "src/core.rs"));
        graph.add_edge("A", "B", calls_edge(1.0));
        graph.add_edge("B", "C", calls_edge(1.0));
        graph.add_edge("C", "A", calls_edge(1.0)); // Cycle back

        let mut metrics = HashMap::new();
        metrics.insert("A".into(), node_metrics(1, 1, 0));
        metrics.insert("B".into(), node_metrics(1, 1, 0));
        metrics.insert("C".into(), node_metrics(1, 1, 0));

        let config = ProcessConfig {
            min_steps: 2,
            ..Default::default()
        };

        let traces = bfs_trace_single(&graph, "A", &config, &metrics);
        // Should not loop forever; trace is A → B → C (C's edge back to A is skipped)
        assert!(!traces.is_empty());
        assert_eq!(traces[0].steps.len(), 3);
        assert!(
            !traces[0].steps[1..].contains(&"A".to_string()),
            "should not revisit A"
        );
    }

    #[test]
    fn test_bfs_max_branching() {
        // A → B, C, D, E, F (5 callees) with max_branching = 2
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("A", "dispatch", "src/main.rs"));
        for ch in ['B', 'C', 'D', 'E', 'F'] {
            let id = ch.to_string();
            graph.add_node(func_node(&id, &format!("handler_{}", ch), "src/h.rs"));
            graph.add_edge("A", &id, calls_edge(1.0));
        }

        let mut metrics = HashMap::new();
        metrics.insert("A".into(), node_metrics(0, 5, 0));
        // Give B highest PageRank, then C
        metrics.insert(
            "B".into(),
            NodeMetrics {
                pagerank: 0.5,
                ..node_metrics(1, 0, 0)
            },
        );
        metrics.insert(
            "C".into(),
            NodeMetrics {
                pagerank: 0.4,
                ..node_metrics(1, 0, 0)
            },
        );
        metrics.insert(
            "D".into(),
            NodeMetrics {
                pagerank: 0.1,
                ..node_metrics(1, 0, 0)
            },
        );
        metrics.insert(
            "E".into(),
            NodeMetrics {
                pagerank: 0.05,
                ..node_metrics(1, 0, 0)
            },
        );
        metrics.insert(
            "F".into(),
            NodeMetrics {
                pagerank: 0.01,
                ..node_metrics(1, 0, 0)
            },
        );

        let config = ProcessConfig {
            max_branching: 2,
            min_steps: 2,
            ..Default::default()
        };

        let traces = bfs_trace_single(&graph, "A", &config, &metrics);
        // Should only follow top 2 by PageRank: B and C
        assert_eq!(traces.len(), 2);
        let terminal_ids: HashSet<String> = traces.iter().map(|t| t.terminal_id.clone()).collect();
        assert!(terminal_ids.contains("B"));
        assert!(terminal_ids.contains("C"));
    }

    #[test]
    fn test_bfs_max_depth() {
        // Long chain: 0 → 1 → 2 → ... → 9 with max_depth = 3
        let mut graph = CodeGraph::new();
        let mut metrics = HashMap::new();
        for i in 0..10 {
            let id = format!("n{}", i);
            graph.add_node(func_node(&id, &format!("step_{}", i), "src/chain.rs"));
            metrics.insert(id.clone(), node_metrics(if i == 0 { 0 } else { 1 }, 1, 0));
            if i > 0 {
                graph.add_edge(&format!("n{}", i - 1), &id, calls_edge(1.0));
            }
        }

        let config = ProcessConfig {
            max_trace_depth: 3,
            min_steps: 2,
            ..Default::default()
        };

        let traces = bfs_trace_single(&graph, "n0", &config, &metrics);
        // Trace should be truncated at depth 3: [n0, n1, n2, n3]
        assert!(!traces.is_empty());
        assert!(traces[0].steps.len() <= 4); // depth 3 = max 4 nodes
    }

    // ── Deduplication Tests ─────────────────────────────────────

    #[test]
    fn test_dedup_subset() {
        let mut traces = vec![
            ProcessTrace {
                entry_point_id: "A".into(),
                terminal_id: "D".into(),
                steps: vec!["A".into(), "B".into(), "C".into(), "D".into()],
                community_ids: HashSet::new(),
            },
            ProcessTrace {
                entry_point_id: "A".into(),
                terminal_id: "C".into(),
                steps: vec!["A".into(), "B".into(), "C".into()], // Subset of first
                community_ids: HashSet::new(),
            },
            ProcessTrace {
                entry_point_id: "X".into(),
                terminal_id: "Z".into(),
                steps: vec!["X".into(), "Y".into(), "Z".into()], // Not a subset
                community_ids: HashSet::new(),
            },
        ];

        deduplicate_subset(&mut traces);
        assert_eq!(traces.len(), 2);
    }

    #[test]
    fn test_dedup_endpoints() {
        let mut traces = vec![
            ProcessTrace {
                entry_point_id: "A".into(),
                terminal_id: "D".into(),
                steps: vec!["A".into(), "B".into(), "D".into()], // Shorter
                community_ids: HashSet::new(),
            },
            ProcessTrace {
                entry_point_id: "A".into(),
                terminal_id: "D".into(),
                steps: vec!["A".into(), "B".into(), "C".into(), "D".into()], // Longer
                community_ids: HashSet::new(),
            },
        ];

        deduplicate_endpoints(&mut traces);
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].steps.len(), 4); // Kept the longer one
    }

    // ── Classification Tests ────────────────────────────────────

    #[test]
    fn test_classify_intra_community() {
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("A", "start", "src/a.rs"));

        let trace = ProcessTrace {
            entry_point_id: "A".into(),
            terminal_id: "C".into(),
            steps: vec!["A".into(), "B".into(), "C".into()],
            community_ids: [0].into_iter().collect(),
        };

        let processes = classify_processes(vec![trace], &graph);
        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].process_type, ProcessType::IntraCommunity);
    }

    #[test]
    fn test_classify_cross_community() {
        let mut graph = CodeGraph::new();
        graph.add_node(func_node("A", "main", "src/a.rs"));

        let trace = ProcessTrace {
            entry_point_id: "A".into(),
            terminal_id: "C".into(),
            steps: vec!["A".into(), "B".into(), "C".into()],
            community_ids: [0, 1].into_iter().collect(),
        };

        let processes = classify_processes(vec![trace], &graph);
        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].process_type, ProcessType::CrossCommunity);
    }
}
