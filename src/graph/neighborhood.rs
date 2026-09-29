//! Entity neighbourhood ("ego-graph") around any node of the knowledge graph.
//!
//! Serves `GET /api/graph/neighborhood` and the agent-facing tools: given a
//! centre entity, return the nodes reachable within `depth` hops over the
//! selected layers, keeping the strongest edges first, bounded by `limit`.
//!
//! The work is split in two so that everything that decides *what* is
//! returned is testable without Neo4j:
//!
//! 1. **Fetch** — [`crate::neo4j::GraphStore::get_entity_neighborhood`]
//!    walks the graph hop by hop and returns a [`RawNeighborhood`]: a
//!    bounded candidate set. Each expanded node contributes at most
//!    `fanout` edges (its strongest ones), each hop expands at most
//!    `frontier_cap` nodes, and container nodes (project, workspace) are
//!    never expanded unless they are the centre. So a hub such as `lib.rs`
//!    or a project cannot blow up the query. [`expand_in_memory`] is the
//!    same walk over an [`InMemoryGraph`] (used by the mock store and tests).
//! 2. **Select** — [`select_neighborhood`] is a pure function over that
//!    candidate set: it applies `min_weight` and the layer filter, runs a BFS
//!    from the centre, admits nodes hop by hop strongest-edge-first until
//!    `limit`, and computes `truncated` and the stats.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// ============================================================================
// Bounds
// ============================================================================

/// Default BFS depth.
pub const DEFAULT_DEPTH: u32 = 2;
/// Maximum BFS depth.
pub const MAX_DEPTH: u32 = 3;
/// Default max number of nodes returned (centre included).
pub const DEFAULT_LIMIT: usize = 120;
/// Hard cap on the number of nodes returned.
pub const MAX_LIMIT: usize = 400;
/// Max edges followed out of a single node per hop (strongest first).
pub const DEFAULT_FANOUT: usize = 30;

// ============================================================================
// Layers
// ============================================================================

/// A slice of the graph a relationship belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    /// Source code structure and history of files.
    Code,
    /// Notes, decisions, documents and what they are attached to.
    Knowledge,
    /// Projects, plans, tasks, steps, milestones, releases, commits.
    Planning,
    /// Learned associations: synapses and emergent skills.
    Neural,
    /// Agents at work: personas, protocols, chat sessions.
    Behavioral,
}

impl Layer {
    pub const ALL: [Layer; 5] = [
        Layer::Code,
        Layer::Knowledge,
        Layer::Planning,
        Layer::Neural,
        Layer::Behavioral,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Layer::Code => "code",
            Layer::Knowledge => "knowledge",
            Layer::Planning => "planning",
            Layer::Neural => "neural",
            Layer::Behavioral => "behavioral",
        }
    }

    pub fn parse(s: &str) -> Option<Layer> {
        Layer::ALL.into_iter().find(|l| l.as_str() == s)
    }

    /// Parse a comma-separated layer list. `None`/empty means all layers.
    pub fn parse_csv(csv: Option<&str>) -> Result<Vec<Layer>, String> {
        let Some(csv) = csv.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(Layer::ALL.to_vec());
        };
        let mut out = Vec::new();
        for part in csv.split(',').map(|p| p.trim().to_lowercase()) {
            if part.is_empty() {
                continue;
            }
            let layer = Layer::parse(&part).ok_or_else(|| {
                format!(
                    "Unknown layer '{}'. Valid layers: code, knowledge, planning, neural, behavioral",
                    part
                )
            })?;
            if !out.contains(&layer) {
                out.push(layer);
            }
        }
        if out.is_empty() {
            return Ok(Layer::ALL.to_vec());
        }
        out.sort();
        Ok(out)
    }
}

/// Relationship type → layer, THE single mapping used by the neighbourhood.
///
/// Each row is `(relationship type, source node type or None for "any",
/// layer)`; the first matching row wins, so qualified rows come before the
/// catch-all row of the same type. A relationship type absent from this table
/// is never traversed. Only two types need a qualifier: `EXTENDS` (struct
/// inheritance is code, persona inheritance is behavioural) and `BELONGS_TO`
/// (its meaning follows the kind of thing that belongs to the project).
///
/// Deliberately absent: plumbing and high-volume edges that carry no meaning
/// for a reader (`HAS_EVENT`, `HAS_IMPORT`/`IMPORTS_SYMBOL` through `Import`
/// nodes, `IMPLEMENTS_FOR`/`IMPLEMENTS_TRAIT` through `Impl` nodes, trajectory
/// and trigger bookkeeping, `HAS_CHUNK`).
pub const REL_LAYERS: &[(&str, Option<&str>, Layer)] = &[
    // --- code: structure, dependencies, history of files ---
    ("CONTAINS", None, Layer::Code), // project→file, file→symbol
    ("IMPORTS", None, Layer::Code),
    ("CALLS", None, Layer::Code),
    ("EXTENDS", Some("persona"), Layer::Behavioral),
    ("EXTENDS", None, Layer::Code),
    ("IMPLEMENTS", None, Layer::Code),
    ("CO_CHANGED", None, Layer::Code),
    ("INCLUDES_ENTITY", None, Layer::Code), // feature_graph→code
    ("TOUCHES", None, Layer::Code),         // commit→file
    ("BELONGS_TO", Some("feature_graph"), Layer::Code),
    // --- knowledge: notes, decisions, documents and their anchors ---
    ("LINKED_TO", None, Layer::Knowledge),
    ("AFFECTS", None, Layer::Knowledge),
    ("SUPERSEDES", None, Layer::Knowledge),
    ("HAS_NOTE", None, Layer::Knowledge),
    ("HAS_DOCUMENT", None, Layer::Knowledge),
    // --- planning: work breakdown and its outcomes ---
    ("HAS_PLAN", None, Layer::Planning),
    ("HAS_TASK", None, Layer::Planning),
    ("HAS_STEP", None, Layer::Planning),
    ("DEPENDS_ON", None, Layer::Planning),
    ("INFORMED_BY", None, Layer::Planning), // task→decision
    ("RESULTED_IN", None, Layer::Planning), // plan→commit
    ("RESOLVED_BY", None, Layer::Planning), // task→commit
    ("TARGETS_MILESTONE", None, Layer::Planning),
    ("INCLUDES_TASK", None, Layer::Planning),
    ("HAS_MILESTONE", None, Layer::Planning),
    ("HAS_WORKSPACE_MILESTONE", None, Layer::Planning),
    ("HAS_RELEASE", None, Layer::Planning),
    ("INCLUDES_COMMIT", None, Layer::Planning),
    ("CONSTRAINED_BY", None, Layer::Planning),
    ("BELONGS_TO_WORKSPACE", None, Layer::Planning),
    // --- neural: learned associations ---
    ("SYNAPSE", None, Layer::Neural),
    ("MEMBER_OF", None, Layer::Neural),       // note→skill
    ("MEMBER_OF_SKILL", None, Layer::Neural), // decision→skill
    ("BELONGS_TO", Some("skill"), Layer::Neural),
    // --- behavioral: agents, protocols, conversations ---
    ("KNOWS", None, Layer::Behavioral), // persona→file
    ("USES", None, Layer::Behavioral),  // persona→note/decision
    ("MASTERS", None, Layer::Behavioral),
    ("FOLLOWS", None, Layer::Behavioral),
    ("BELONGS_TO_SKILL", None, Layer::Behavioral), // protocol→skill
    ("HAS_STATE", None, Layer::Behavioral),
    ("DISCUSSED", None, Layer::Behavioral), // chat_session→code
    ("ASSOCIATED_WITH", None, Layer::Behavioral), // chat_session→task/plan
    ("HAS_CHAT_SESSION", None, Layer::Behavioral),
    ("BELONGS_TO", Some("persona"), Layer::Behavioral),
    ("BELONGS_TO", Some("protocol"), Layer::Behavioral),
];

/// Layer of a relationship, given the type of its *source* (start) node.
pub fn layer_of(rel: &str, source_type: &str) -> Option<Layer> {
    REL_LAYERS
        .iter()
        .find(|(r, src, _)| *r == rel && src.is_none_or(|s| s == source_type))
        .map(|(_, _, l)| *l)
}

/// Relationship types that may carry an edge of one of `layers`.
pub fn rel_types_for(layers: &[Layer]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = REL_LAYERS
        .iter()
        .filter(|(_, _, l)| layers.contains(l))
        .map(|(r, _, _)| *r)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Edge weight formula per relationship type (Cypher, `r` = relationship).
///
/// | rel         | weight                                   |
/// |-------------|------------------------------------------|
/// | SYNAPSE     | `r.weight` (0–1, Hebbian strength)       |
/// | KNOWS, USES | `r.weight` (0–1, persona affinity)       |
/// | CO_CHANGED  | `count / (count + 10)` (9 → 0.47)        |
/// | DISCUSSED   | `mention_count / (mention_count + 5)`    |
/// | CALLS       | `r.confidence` (0.3–0.9, call resolution)|
/// | LINKED_TO   | `r.similarity_score` for propagated links, else 1.0 |
/// | anything else | 1.0 (structural fact)                  |
///
/// Results are clamped to 0–1.
pub const EDGE_WEIGHT_CYPHER: &str = "CASE type(r) \
     WHEN 'SYNAPSE' THEN coalesce(toFloat(r.weight), 0.5) \
     WHEN 'KNOWS' THEN coalesce(toFloat(r.weight), 0.5) \
     WHEN 'USES' THEN coalesce(toFloat(r.weight), 0.5) \
     WHEN 'CO_CHANGED' THEN toFloat(coalesce(r.count, 0)) / (coalesce(r.count, 0) + 10.0) \
     WHEN 'DISCUSSED' THEN toFloat(coalesce(r.mention_count, 1)) / (coalesce(r.mention_count, 1) + 5.0) \
     WHEN 'CALLS' THEN coalesce(toFloat(r.confidence), 1.0) \
     WHEN 'LINKED_TO' THEN coalesce(toFloat(r.similarity_score), 1.0) \
     ELSE 1.0 END";

// ============================================================================
// Entity types
// ============================================================================

/// One row of the entity-type table.
#[derive(Debug, Clone, Copy)]
pub struct EntityKind {
    /// API name (snake_case), e.g. `feature_graph`.
    pub api: &'static str,
    /// Neo4j label, e.g. `FeatureGraph`.
    pub label: &'static str,
    /// Property holding the public id.
    pub id_prop: &'static str,
    /// Containers are never expanded unless they are the centre: every note
    /// of a project links to it, so walking through one reaches everything.
    pub container: bool,
}

const fn kind(api: &'static str, label: &'static str, id_prop: &'static str) -> EntityKind {
    EntityKind {
        api,
        label,
        id_prop,
        container: false,
    }
}

/// Node types that can appear in a neighbourhood (and be its centre).
/// Labels not listed here (Import, Impl, ChatEvent, DocumentChunk, …) are
/// never returned.
pub const ENTITY_KINDS: &[EntityKind] = &[
    kind("note", "Note", "id"),
    kind("decision", "Decision", "id"),
    kind("constraint", "Constraint", "id"),
    kind("task", "Task", "id"),
    kind("plan", "Plan", "id"),
    kind("step", "Step", "id"),
    kind("milestone", "Milestone", "id"),
    kind("workspace_milestone", "WorkspaceMilestone", "id"),
    kind("release", "Release", "id"),
    EntityKind {
        container: true,
        ..kind("project", "Project", "id")
    },
    EntityKind {
        container: true,
        ..kind("workspace", "Workspace", "id")
    },
    kind("file", "File", "path"),
    kind("function", "Function", "id"),
    kind("struct", "Struct", "id"),
    kind("trait", "Trait", "id"),
    kind("enum", "Enum", "id"),
    kind("skill", "Skill", "id"),
    kind("persona", "Persona", "id"),
    kind("protocol", "Protocol", "id"),
    kind("protocol_state", "ProtocolState", "id"),
    kind("feature_graph", "FeatureGraph", "id"),
    kind("commit", "Commit", "hash"),
    kind("chat_session", "ChatSession", "id"),
    kind("document", "Document", "id"),
];

/// Look up an entity kind by its API name (`feature_graph`, `chat_session`…).
pub fn entity_kind(api: &str) -> Option<&'static EntityKind> {
    ENTITY_KINDS.iter().find(|k| k.api == api)
}

/// Map Neo4j labels to the first known entity kind.
pub fn kind_for_labels<S: AsRef<str>>(labels: &[S]) -> Option<&'static EntityKind> {
    labels
        .iter()
        .find_map(|l| ENTITY_KINDS.iter().find(|k| k.label == l.as_ref()))
}

/// All Neo4j labels a neighbour may carry.
pub fn all_labels() -> Vec<&'static str> {
    ENTITY_KINDS.iter().map(|k| k.label).collect()
}

// ============================================================================
// Node properties → label, subtitle, salience
// ============================================================================

/// The handful of node properties the neighbourhood reads. Every field is
/// optional: each node type fills only a few.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NodeProps {
    pub id: Option<String>,
    pub name: Option<String>,
    pub title: Option<String>,
    pub slug: Option<String>,
    pub path: Option<String>,
    pub hash: Option<String>,
    pub filename: Option<String>,
    pub format: Option<String>,
    pub content: Option<String>,
    pub description: Option<String>,
    pub message: Option<String>,
    pub preview: Option<String>,
    pub note_type: Option<String>,
    pub importance: Option<String>,
    pub status: Option<String>,
    pub priority: Option<f64>,
    pub energy: Option<f64>,
    pub pagerank: Option<f64>,
    pub message_count: Option<f64>,
    pub model: Option<String>,
    pub file_path: Option<String>,
    pub line_start: Option<i64>,
    pub state_type: Option<String>,
    pub constraint_type: Option<String>,
    pub protocol_category: Option<String>,
    pub author: Option<String>,
}

fn norm_status(s: Option<&str>) -> String {
    s.unwrap_or_default().to_lowercase().replace(['_', ' '], "")
}

/// `log10(pagerank)` mapped from [1e-5, 1] to [0, 1]. PageRank is heavily
/// skewed (median file ≈ 3e-4, max 1.0) so a linear scale would flatten
/// everything but the top few hubs.
pub fn pagerank_salience(pr: f64) -> f64 {
    if pr <= 0.0 || !pr.is_finite() {
        return 0.0;
    }
    ((pr.log10() + 5.0) / 5.0).clamp(0.0, 1.0)
}

/// A node's own salience in 0–1 (the `weight` of a returned node).
///
/// | type                         | salience                                         |
/// |------------------------------|--------------------------------------------------|
/// | note, skill, persona         | `energy` (neural activation, already 0–1)        |
/// | file, function               | PageRank, log-scaled (see [`pagerank_salience`]) |
/// | struct, trait, enum          | PageRank if computed, else 0.3                   |
/// | decision                     | status: accepted 0.8, proposed 0.6, other 0.2    |
/// | task, plan, step             | ½·status activity + ½·priority/100               |
/// | milestone, release, ws milestone | open/in progress 0.8, done 0.4               |
/// | project, workspace           | 1.0                                              |
/// | chat_session                 | `messages / (messages + 50)`                     |
/// | protocol 0.6 · feature_graph, document, constraint 0.5 · protocol_state 0.4 · commit 0.3 |
pub fn node_salience(node_type: &str, p: &NodeProps) -> f64 {
    let status = norm_status(p.status.as_deref());
    let priority = (p.priority.unwrap_or(50.0) / 100.0).clamp(0.0, 1.0);
    let w = match node_type {
        "note" | "skill" | "persona" => p.energy.unwrap_or(0.5),
        "file" | "function" => p.pagerank.map(pagerank_salience).unwrap_or(0.2),
        "struct" | "trait" | "enum" => p.pagerank.map(pagerank_salience).unwrap_or(0.3),
        "decision" => match status.as_str() {
            "accepted" => 0.8,
            "proposed" => 0.6,
            _ => 0.2,
        },
        "task" | "step" => {
            let activity = match status.as_str() {
                "inprogress" => 1.0,
                "blocked" => 0.9,
                "pending" => 0.7,
                "failed" => 0.5,
                "completed" => 0.3,
                _ => 0.4,
            };
            0.5 * activity + 0.5 * priority
        }
        "plan" => {
            let activity = match status.as_str() {
                "inprogress" => 1.0,
                "approved" => 0.8,
                "draft" => 0.6,
                "completed" => 0.3,
                _ => 0.1,
            };
            0.5 * activity + 0.5 * priority
        }
        "milestone" | "workspace_milestone" | "release" => match status.as_str() {
            "open" | "inprogress" | "planned" => 0.8,
            _ => 0.4,
        },
        "project" | "workspace" => 1.0,
        "chat_session" => {
            let m = p.message_count.unwrap_or(0.0).max(0.0);
            m / (m + 50.0)
        }
        "protocol" => 0.6,
        "protocol_state" => 0.4,
        "commit" => 0.3,
        _ => 0.5,
    };
    if w.is_finite() {
        w.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// First non-empty line, markdown heading marks stripped, cut at `max` chars.
fn headline(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .unwrap_or("");
    truncate(line, max)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{}…", cut.trim_end())
    }
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn join_parts(parts: &[Option<String>]) -> Option<String> {
    let v: Vec<&str> = parts
        .iter()
        .filter_map(|p| p.as_deref())
        .filter(|s| !s.is_empty())
        .collect();
    (!v.is_empty()).then(|| v.join(" · "))
}

/// Short human title and optional one-liner for a node.
pub fn node_label(node_type: &str, public_id: &str, p: &NodeProps) -> (String, Option<String>) {
    const MAX: usize = 80;
    let status = p.status.clone().map(|s| s.to_lowercase());
    let first = |cands: &[&Option<String>]| {
        cands
            .iter()
            .filter_map(|c| c.as_deref())
            .map(|s| headline(s, MAX))
            .find(|s| !s.is_empty())
    };
    let (label, subtitle) = match node_type {
        "note" => (
            first(&[&p.content]),
            join_parts(&[p.note_type.clone(), p.importance.clone(), status]),
        ),
        "decision" => (first(&[&p.description]), join_parts(&[status])),
        "constraint" => (
            first(&[&p.description]),
            join_parts(&[p.constraint_type.clone()]),
        ),
        "task" | "plan" | "milestone" | "workspace_milestone" | "release" | "step" => (
            first(&[&p.title, &p.description]),
            join_parts(&[status, p.priority.map(|x| format!("priority {}", x as i64))]),
        ),
        "project" | "workspace" => (first(&[&p.name]), p.slug.clone()),
        "file" => (
            p.path.as_deref().map(|s| basename(s).to_string()),
            p.path.clone(),
        ),
        "function" | "struct" | "trait" | "enum" => (
            first(&[&p.name]),
            p.file_path.as_ref().map(|f| match p.line_start {
                Some(l) => format!("{}:{}", basename(f), l),
                None => basename(f).to_string(),
            }),
        ),
        "skill" | "persona" => (
            first(&[&p.name]),
            join_parts(&[status, p.description.as_deref().map(|d| headline(d, MAX))]),
        ),
        "protocol" => (
            first(&[&p.name]),
            join_parts(&[p.protocol_category.clone()]),
        ),
        "protocol_state" => (first(&[&p.name]), join_parts(&[p.state_type.clone()])),
        "feature_graph" => (
            first(&[&p.name]),
            p.description.as_deref().map(|d| headline(d, MAX)),
        ),
        "commit" => (
            first(&[&p.message]),
            join_parts(&[
                p.hash.as_deref().map(|h| h.chars().take(8).collect()),
                p.author.clone(),
            ]),
        ),
        "chat_session" => (
            first(&[&p.title, &p.preview]),
            join_parts(&[
                p.model.clone(),
                p.message_count.map(|m| format!("{} messages", m as i64)),
            ]),
        ),
        "document" => (first(&[&p.filename]), p.format.clone()),
        _ => (first(&[&p.name, &p.title]), None),
    };
    let label = label
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| truncate(public_id, MAX));
    (label, subtitle)
}

// ============================================================================
// Raw candidate set (output of the fetch)
// ============================================================================

/// A candidate node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    /// Salience, 0–1 (see [`node_salience`]).
    pub weight: f64,
}

impl RawNode {
    pub fn from_props(node_type: &str, public_id: String, props: &NodeProps) -> Self {
        let (label, subtitle) = node_label(node_type, &public_id, props);
        RawNode {
            weight: node_salience(node_type, props),
            node_type: node_type.to_string(),
            id: public_id,
            label,
            subtitle,
        }
    }
}

/// A candidate edge (direction as stored in the graph).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawEdge {
    pub source: String,
    pub target: String,
    pub rel: String,
    pub weight: f64,
    pub layer: Layer,
}

/// Bounded candidate set around a centre, produced by the store.
#[derive(Debug, Clone, Default)]
pub struct RawNeighborhood {
    pub center: Option<RawNode>,
    pub nodes: Vec<RawNode>,
    pub edges: Vec<RawEdge>,
}

/// Parameters of a neighbourhood query (already validated and clamped).
#[derive(Debug, Clone)]
pub struct NeighborhoodParams {
    pub depth: u32,
    pub min_weight: f64,
    pub limit: usize,
    pub layers: Vec<Layer>,
    /// Max edges followed out of one node per hop. The centre itself gets
    /// `max(fanout, limit)` (see [`NeighborhoodParams::fanout_for_hop`]).
    pub fanout: usize,
    /// Max nodes expanded per hop.
    pub frontier_cap: usize,
}

impl NeighborhoodParams {
    /// Clamp raw query values into their allowed ranges.
    pub fn clamped(
        depth: Option<u32>,
        min_weight: Option<f64>,
        limit: Option<usize>,
        layers: Vec<Layer>,
    ) -> Self {
        let limit = limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
        let min_weight = min_weight
            .filter(|w| w.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        NeighborhoodParams {
            depth: depth.unwrap_or(DEFAULT_DEPTH).clamp(1, MAX_DEPTH),
            min_weight,
            limit,
            layers,
            fanout: DEFAULT_FANOUT,
            frontier_cap: limit,
        }
    }
}

impl NeighborhoodParams {
    /// Fan-out used at `hop` (1-based): the centre may contribute up to
    /// `limit` neighbours so depth 1 is never capped below what can be
    /// returned; deeper hops use `fanout`.
    pub fn fanout_for_hop(&self, hop: u32) -> usize {
        if hop == 1 {
            self.fanout.max(self.limit)
        } else {
            self.fanout
        }
    }
}

/// Deterministic "strongest first" ordering: edge weight desc, then the
/// neighbour's salience desc, then its id asc.
fn strongest_first(a: (f64, f64, &str), b: (f64, f64, &str)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0)
        .then(b.1.total_cmp(&a.1))
        .then_with(|| a.2.cmp(b.2))
}

/// Rank the nodes discovered at one hop and keep the `cap` best for the next
/// expansion. `best` maps node id → (best incoming edge weight, salience).
pub fn next_frontier(best: &HashMap<String, (f64, f64)>, cap: usize) -> Vec<String> {
    let mut v: Vec<(&String, &(f64, f64))> = best.iter().collect();
    v.sort_by(|a, b| strongest_first((a.1 .0, a.1 .1, a.0), (b.1 .0, b.1 .1, b.0)));
    v.into_iter().take(cap).map(|(id, _)| id.clone()).collect()
}

// ============================================================================
// In-memory graph (mock store + tests)
// ============================================================================

/// A small in-memory graph with the same shape as the fetch output.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraph {
    pub nodes: HashMap<String, RawNode>,
    pub edges: Vec<RawEdge>,
}

impl InMemoryGraph {
    pub fn add_node(&mut self, node: RawNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_edge(&mut self, source: &str, target: &str, rel: &str, weight: f64) {
        let src_type = self
            .nodes
            .get(source)
            .map(|n| n.node_type.clone())
            .unwrap_or_default();
        let layer = layer_of(rel, &src_type).unwrap_or(Layer::Knowledge);
        self.edges.push(RawEdge {
            source: source.to_string(),
            target: target.to_string(),
            rel: rel.to_string(),
            weight,
            layer,
        });
    }
}

/// The bounded hop-by-hop walk over an [`InMemoryGraph`] — same rules as the
/// Neo4j fetch: per-node fan-out cap, per-hop frontier cap, containers are
/// leaves, `min_weight` and layers applied while walking.
/// Returns `None` if the centre does not exist (or has another type).
pub fn expand_in_memory(
    graph: &InMemoryGraph,
    center_type: &str,
    center_id: &str,
    params: &NeighborhoodParams,
) -> Option<RawNeighborhood> {
    let center = graph
        .nodes
        .get(center_id)
        .filter(|n| n.node_type == center_type)?
        .clone();

    let mut adj: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, e) in graph.edges.iter().enumerate() {
        if !params.layers.contains(&e.layer) || e.weight < params.min_weight {
            continue;
        }
        adj.entry(e.source.as_str()).or_default().push(i);
        if e.source != e.target {
            adj.entry(e.target.as_str()).or_default().push(i);
        }
    }

    let mut seen: HashSet<String> = HashSet::from([center.id.clone()]);
    let mut nodes = Vec::new();
    let mut edges: Vec<RawEdge> = Vec::new();
    let mut edge_seen: HashSet<usize> = HashSet::new();
    let mut frontier = vec![center.id.clone()];

    for hop in 1..=params.depth {
        let mut best: HashMap<String, (f64, f64)> = HashMap::new();
        for fid in &frontier {
            let is_center = *fid == center.id;
            let Some(fnode) = graph.nodes.get(fid) else {
                continue;
            };
            if !is_center && entity_kind(&fnode.node_type).is_some_and(|k| k.container) {
                continue;
            }
            let mut cands: Vec<(usize, &RawNode)> = adj
                .get(fid.as_str())
                .into_iter()
                .flatten()
                .filter_map(|&i| {
                    let e = &graph.edges[i];
                    let other = if e.source == *fid { &e.target } else { &e.source };
                    graph.nodes.get(other).map(|n| (i, n))
                })
                .collect();
            cands.sort_by(|a, b| {
                strongest_first(
                    (graph.edges[a.0].weight, a.1.weight, &a.1.id),
                    (graph.edges[b.0].weight, b.1.weight, &b.1.id),
                )
            });
            for (i, other) in cands.into_iter().take(params.fanout_for_hop(hop)) {
                if edge_seen.insert(i) {
                    edges.push(graph.edges[i].clone());
                }
                if !seen.contains(&other.id) {
                    let w = graph.edges[i].weight;
                    let entry = best.entry(other.id.clone()).or_insert((w, other.weight));
                    if w > entry.0 {
                        entry.0 = w;
                    }
                }
            }
        }
        for id in best.keys() {
            seen.insert(id.clone());
            nodes.push(graph.nodes[id].clone());
        }
        frontier = next_frontier(&best, params.frontier_cap);
        if frontier.is_empty() {
            break;
        }
    }

    Some(RawNeighborhood {
        center: Some(center),
        nodes,
        edges,
    })
}

// ============================================================================
// Selection (pure)
// ============================================================================

/// A node of the response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeighborhoodNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    pub weight: f64,
    pub depth: u32,
    /// Layer of the edge through which the node was reached (the centre
    /// takes the layer of its type's home: see [`home_layer`]).
    pub layer: Layer,
}

/// An edge of the response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeighborhoodEdge {
    pub source: String,
    pub target: String,
    pub rel: String,
    pub weight: f64,
    pub layer: Layer,
}

/// Reference to the centre.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CenterRef {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
}

/// Counters over the neighbourhood.
///
/// | field                | layers filter | min_weight | limit  |
/// |----------------------|---------------|------------|--------|
/// | `by_type`, `by_rel`  | after         | after      | after (= returned nodes / edges) |
/// | `total_before_limit` | after         | after      | before |
/// | `by_layer`           | **before**    | after      | before |
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeighborhoodStats {
    pub by_type: BTreeMap<String, usize>,
    pub by_rel: BTreeMap<String, usize>,
    /// Nodes reachable within `depth` over the kept edges, before `limit`
    /// (centre included). Bounded by the fetch's fan-out caps, so on hubs it
    /// is a lower bound of the true neighbourhood size.
    pub total_before_limit: usize,
    /// Edges per layer in the candidate neighbourhood walked over ALL layers
    /// (same centre, depth and min_weight), before the `layers` filter and
    /// before `limit` — so a UI can badge layer toggles that are off. Every
    /// layer is present, with 0 when empty.
    pub by_layer: BTreeMap<String, usize>,
}

/// Response of `GET /api/graph/neighborhood`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeighborhoodResponse {
    pub center: CenterRef,
    pub nodes: Vec<NeighborhoodNode>,
    pub edges: Vec<NeighborhoodEdge>,
    pub truncated: bool,
    pub stats: NeighborhoodStats,
    /// Echo of the effective (clamped) parameters.
    pub params: EffectiveParams,
}

/// Parameters actually applied, after clamping.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectiveParams {
    pub depth: u32,
    pub min_weight: f64,
    pub limit: usize,
    pub layers: Vec<Layer>,
}

/// The layer a node type "lives" in — used for the centre node, which is not
/// reached through any edge.
pub fn home_layer(node_type: &str) -> Layer {
    match node_type {
        "file" | "function" | "struct" | "trait" | "enum" | "feature_graph" => Layer::Code,
        "note" | "decision" | "constraint" | "document" => Layer::Knowledge,
        "skill" => Layer::Neural,
        "persona" | "protocol" | "protocol_state" | "chat_session" => Layer::Behavioral,
        _ => Layer::Planning,
    }
}

/// Select the returned neighbourhood from a candidate set.
///
/// 1. Keep edges whose layer is selected and whose weight ≥ `min_weight`
///    (duplicates by `(source, target, rel)` collapse).
/// 2. BFS from the centre over those edges (undirected) up to `depth`:
///    this gives `total_before_limit`.
/// 3. Admit nodes hop by hop: at hop *d*, candidates are the unadmitted
///    neighbours of nodes admitted at hop *d-1*, ranked by their strongest
///    such edge (weight desc), then their salience desc, then id asc; they
///    are admitted until `limit` nodes (centre included) are reached. Nodes
///    left out are not expanded further.
/// 4. Return every kept edge whose two ends were admitted.
pub fn select_neighborhood(
    raw: &RawNeighborhood,
    params: &NeighborhoodParams,
) -> Option<NeighborhoodResponse> {
    let center = raw.center.as_ref()?;
    let mut node_by_id: HashMap<&str, &RawNode> =
        raw.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    node_by_id.insert(center.id.as_str(), center);

    // 1. edge filter + dedup
    let mut seen_keys: HashSet<(&str, &str, &str)> = HashSet::new();
    let edges: Vec<&RawEdge> = raw
        .edges
        .iter()
        .filter(|e| params.layers.contains(&e.layer) && e.weight >= params.min_weight)
        .filter(|e| node_by_id.contains_key(e.source.as_str()))
        .filter(|e| node_by_id.contains_key(e.target.as_str()))
        .filter(|e| seen_keys.insert((&e.source, &e.target, &e.rel)))
        .collect();

    let mut adj: HashMap<&str, Vec<&RawEdge>> = HashMap::new();
    for e in &edges {
        adj.entry(e.source.as_str()).or_default().push(e);
        if e.source != e.target {
            adj.entry(e.target.as_str()).or_default().push(e);
        }
    }
    let other_end = |e: &RawEdge, from: &str| -> String {
        if e.source == from {
            e.target.clone()
        } else {
            e.source.clone()
        }
    };

    // 2. unlimited BFS for total_before_limit
    let mut reach: HashSet<String> = HashSet::from([center.id.clone()]);
    let mut layer_frontier = vec![center.id.clone()];
    for _ in 0..params.depth {
        let mut next = Vec::new();
        for id in &layer_frontier {
            for e in adj.get(id.as_str()).into_iter().flatten() {
                let o = other_end(e, id);
                if reach.insert(o.clone()) {
                    next.push(o);
                }
            }
        }
        layer_frontier = next;
    }
    let total_before_limit = reach.len();

    // 3. limited, strongest-first admission
    let mut admitted: HashMap<String, (u32, Layer)> = HashMap::new();
    admitted.insert(center.id.clone(), (0, home_layer(&center.node_type)));
    let mut frontier = vec![center.id.clone()];
    let mut full = params.limit <= 1;
    for d in 1..=params.depth {
        if full || frontier.is_empty() {
            break;
        }
        // candidate → (best weight, salience, layer of best edge)
        let mut cands: HashMap<String, (f64, f64, Layer)> = HashMap::new();
        for id in &frontier {
            for e in adj.get(id.as_str()).into_iter().flatten() {
                let o = other_end(e, id);
                if admitted.contains_key(&o) {
                    continue;
                }
                let sal = node_by_id[o.as_str()].weight;
                let entry = cands.entry(o).or_insert((e.weight, sal, e.layer));
                if e.weight > entry.0 || (e.weight == entry.0 && e.layer < entry.2) {
                    entry.0 = e.weight;
                    entry.2 = e.layer;
                }
            }
        }
        let mut ranked: Vec<(String, (f64, f64, Layer))> = cands.into_iter().collect();
        ranked.sort_by(|a, b| strongest_first((a.1 .0, a.1 .1, &a.0), (b.1 .0, b.1 .1, &b.0)));
        let mut next = Vec::new();
        for (id, (_, _, layer)) in ranked {
            if admitted.len() >= params.limit {
                full = true;
                break;
            }
            admitted.insert(id.clone(), (d, layer));
            next.push(id);
        }
        frontier = next;
    }

    // 4. output
    let mut nodes: Vec<NeighborhoodNode> = admitted
        .iter()
        .map(|(id, (depth, layer))| {
            let n = node_by_id[id.as_str()];
            NeighborhoodNode {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                label: n.label.clone(),
                subtitle: n.subtitle.clone(),
                weight: n.weight,
                depth: *depth,
                layer: *layer,
            }
        })
        .collect();
    nodes.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then(b.weight.total_cmp(&a.weight))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut out_edges: Vec<NeighborhoodEdge> = edges
        .iter()
        .filter(|e| admitted.contains_key(&e.source) && admitted.contains_key(&e.target))
        .map(|e| NeighborhoodEdge {
            source: e.source.clone(),
            target: e.target.clone(),
            rel: e.rel.clone(),
            weight: e.weight,
            layer: e.layer,
        })
        .collect();
    out_edges.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.rel.cmp(&b.rel))
    });

    let mut by_type = BTreeMap::new();
    for n in &nodes {
        *by_type.entry(n.node_type.clone()).or_insert(0) += 1;
    }
    let mut by_rel = BTreeMap::new();
    for e in &out_edges {
        *by_rel.entry(e.rel.clone()).or_insert(0) += 1;
    }

    Some(NeighborhoodResponse {
        center: CenterRef {
            id: center.id.clone(),
            node_type: center.node_type.clone(),
        },
        truncated: total_before_limit > nodes.len(),
        nodes,
        edges: out_edges,
        stats: NeighborhoodStats {
            by_type,
            by_rel,
            total_before_limit,
            by_layer: layer_counts(raw, params.depth, params.min_weight),
        },
        params: EffectiveParams {
            depth: params.depth,
            min_weight: params.min_weight,
            limit: params.limit,
            layers: params.layers.clone(),
        },
    })
}

/// Edges per layer (all five keys) in a candidate set: edges of weight ≥
/// `min_weight` whose two ends are reachable from the centre within `depth`
/// hops over such edges, whatever their layer. Pass the candidate set
/// fetched over all layers to get counts "before the layers filter".
pub fn layer_counts(raw: &RawNeighborhood, depth: u32, min_weight: f64) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = Layer::ALL
        .iter()
        .map(|l| (l.as_str().to_string(), 0))
        .collect();
    let Some(center) = raw.center.as_ref() else {
        return out;
    };
    let mut seen_keys: HashSet<(&str, &str, &str)> = HashSet::new();
    let edges: Vec<&RawEdge> = raw
        .edges
        .iter()
        .filter(|e| e.weight >= min_weight)
        .filter(|e| seen_keys.insert((&e.source, &e.target, &e.rel)))
        .collect();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in &edges {
        adj.entry(e.source.as_str()).or_default().push(e.target.as_str());
        adj.entry(e.target.as_str()).or_default().push(e.source.as_str());
    }
    let mut reach: HashSet<&str> = HashSet::from([center.id.as_str()]);
    let mut frontier = vec![center.id.as_str()];
    for _ in 0..depth {
        let mut next = Vec::new();
        for id in frontier {
            for o in adj.get(id).into_iter().flatten() {
                if reach.insert(o) {
                    next.push(*o);
                }
            }
        }
        frontier = next;
    }
    for e in edges {
        if reach.contains(e.source.as_str()) && reach.contains(e.target.as_str()) {
            *out.entry(e.layer.as_str().to_string()).or_insert(0) += 1;
        }
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, t: &str, w: f64) -> RawNode {
        RawNode {
            id: id.into(),
            node_type: t.into(),
            label: id.into(),
            subtitle: None,
            weight: w,
        }
    }

    fn params(depth: u32, min_weight: f64, limit: usize) -> NeighborhoodParams {
        NeighborhoodParams::clamped(Some(depth), Some(min_weight), Some(limit), Layer::ALL.to_vec())
    }

    /// n0 ─SYNAPSE .9─ n1 ─SYNAPSE .8─ n3
    ///  │
    ///  ├─SYNAPSE .2─ n2 ─LINKED_TO─ f1 (file)
    ///  └─LINKED_TO─ p (project) ─HAS_NOTE─ n9
    fn sample() -> InMemoryGraph {
        let mut g = InMemoryGraph::default();
        g.add_node(node("n0", "note", 0.5));
        g.add_node(node("n1", "note", 0.4));
        g.add_node(node("n2", "note", 0.9));
        g.add_node(node("n3", "note", 0.1));
        g.add_node(node("f1", "file", 0.3));
        g.add_node(node("p", "project", 1.0));
        g.add_node(node("n9", "note", 0.7));
        g.add_edge("n0", "n1", "SYNAPSE", 0.9);
        g.add_edge("n0", "n2", "SYNAPSE", 0.2);
        g.add_edge("n1", "n3", "SYNAPSE", 0.8);
        g.add_edge("n2", "f1", "LINKED_TO", 1.0);
        g.add_edge("n0", "p", "LINKED_TO", 1.0);
        g.add_edge("p", "n9", "HAS_NOTE", 1.0);
        g
    }

    fn run(g: &InMemoryGraph, center: (&str, &str), p: &NeighborhoodParams) -> NeighborhoodResponse {
        let raw = expand_in_memory(g, center.0, center.1, p).expect("center");
        select_neighborhood(&raw, p).expect("selection")
    }

    fn ids(r: &NeighborhoodResponse) -> Vec<&str> {
        r.nodes.iter().map(|n| n.id.as_str()).collect()
    }

    #[test]
    fn layer_table_has_no_unreachable_rows() {
        // A qualified row must come before the catch-all of the same type.
        for (i, (rel, src, _)) in REL_LAYERS.iter().enumerate() {
            if src.is_some() {
                assert!(
                    !REL_LAYERS[..i].iter().any(|(r, s, _)| r == rel && s.is_none()),
                    "{rel}/{src:?} is shadowed by an earlier catch-all"
                );
            }
        }
        assert_eq!(layer_of("EXTENDS", "persona"), Some(Layer::Behavioral));
        assert_eq!(layer_of("EXTENDS", "struct"), Some(Layer::Code));
        assert_eq!(layer_of("BELONGS_TO", "skill"), Some(Layer::Neural));
        assert_eq!(layer_of("BELONGS_TO", "note"), None);
        assert_eq!(layer_of("HAS_EVENT", "chat_session"), None);
    }

    #[test]
    fn parse_layers_csv() {
        assert_eq!(Layer::parse_csv(None).unwrap(), Layer::ALL.to_vec());
        assert_eq!(Layer::parse_csv(Some("")).unwrap(), Layer::ALL.to_vec());
        assert_eq!(
            Layer::parse_csv(Some("neural, Code,neural")).unwrap(),
            vec![Layer::Code, Layer::Neural]
        );
        assert!(Layer::parse_csv(Some("code,fabric")).is_err());
    }

    #[test]
    fn rel_types_follow_layers() {
        let neural = rel_types_for(&[Layer::Neural]);
        assert!(neural.contains(&"SYNAPSE"));
        assert!(neural.contains(&"BELONGS_TO"));
        assert!(!neural.contains(&"LINKED_TO"));
    }

    #[test]
    fn center_depth_zero_and_bfs_depths() {
        let g = sample();
        let r = run(&g, ("note", "n0"), &params(2, 0.0, 100));
        assert_eq!(r.nodes[0].id, "n0");
        assert_eq!(r.nodes[0].depth, 0);
        let depth = |id: &str| r.nodes.iter().find(|n| n.id == id).map(|n| n.depth);
        assert_eq!(depth("n1"), Some(1));
        assert_eq!(depth("n3"), Some(2));
        assert_eq!(depth("f1"), Some(2));
        assert_eq!(depth("p"), Some(1));
        // project is a container: not expanded when it isn't the centre
        assert_eq!(depth("n9"), None);
        assert!(!r.truncated);
        assert_eq!(r.stats.total_before_limit, r.nodes.len());
    }

    #[test]
    fn container_center_is_expanded() {
        let g = sample();
        let r = run(&g, ("project", "p"), &params(1, 0.0, 100));
        assert!(ids(&r).contains(&"n9"));
        assert!(ids(&r).contains(&"n0"));
    }

    #[test]
    fn depth_one_only_direct_neighbours() {
        let g = sample();
        let r = run(&g, ("note", "n0"), &params(1, 0.0, 100));
        let mut got = ids(&r);
        got.sort();
        assert_eq!(got, vec!["n0", "n1", "n2", "p"]);
    }

    #[test]
    fn min_weight_prunes_edges_and_unreachable_nodes() {
        let g = sample();
        let r = run(&g, ("note", "n0"), &params(2, 0.5, 100));
        let got = ids(&r);
        // n0-n2 (0.2) is dropped, so n2 and f1 (only reachable through it) go too
        assert!(!got.contains(&"n2"));
        assert!(!got.contains(&"f1"));
        assert!(got.contains(&"n3"));
        assert!(r.edges.iter().all(|e| e.weight >= 0.5));
    }

    #[test]
    fn limit_keeps_strongest_and_flags_truncation() {
        let g = sample();
        let r = run(&g, ("note", "n0"), &params(2, 0.0, 3));
        assert_eq!(r.nodes.len(), 3);
        assert!(r.truncated);
        assert_eq!(r.stats.total_before_limit, 6);
        // hop 1 ranked: n0-p 1.0 (sal 1.0), n0-n1 0.9, n0-n2 0.2
        assert_eq!(ids(&r), vec!["n0", "p", "n1"]);
        // only edges between admitted nodes
        assert!(r
            .edges
            .iter()
            .all(|e| ids(&r).contains(&e.source.as_str()) && ids(&r).contains(&e.target.as_str())));
    }

    #[test]
    fn layers_restrict_traversal() {
        let g = sample();
        let p = NeighborhoodParams::clamped(Some(3), None, None, vec![Layer::Neural]);
        let r = run(&g, ("note", "n0"), &p);
        let mut got = ids(&r);
        got.sort();
        assert_eq!(got, vec!["n0", "n1", "n2", "n3"]);
        assert!(r.edges.iter().all(|e| e.layer == Layer::Neural));
    }

    #[test]
    fn by_layer_counts_all_layers_before_filter() {
        let g = sample();
        // candidate set walked over all layers
        let all = expand_in_memory(&g, "note", "n0", &params(2, 0.0, 100)).unwrap();
        let p = NeighborhoodParams::clamped(Some(2), None, None, vec![Layer::Neural]);
        let mut r = select_neighborhood(&expand_in_memory(&g, "note", "n0", &p).unwrap(), &p)
            .unwrap();
        r.stats.by_layer = layer_counts(&all, p.depth, p.min_weight);
        // SYNAPSE n0-n1, n0-n2, n1-n3
        assert_eq!(r.stats.by_layer["neural"], 3);
        // LINKED_TO n2-f1, n0-p (p is a leaf: p-n9 never fetched)
        assert_eq!(r.stats.by_layer["knowledge"], 2);
        assert_eq!(r.stats.by_layer["code"], 0);
        assert_eq!(r.stats.by_layer.len(), 5);
        // by_rel stays after the filter
        assert_eq!(r.stats.by_rel.keys().collect::<Vec<_>>(), vec!["SYNAPSE"]);
        // min_weight applies to by_layer too
        let c = layer_counts(&all, 2, 0.5);
        assert_eq!(c["neural"], 2);
    }

    #[test]
    fn fanout_caps_hub_expansion() {
        let mut g = InMemoryGraph::default();
        g.add_node(node("hub", "file", 1.0));
        for i in 0..100 {
            let id = format!("n{i:03}");
            g.add_node(node(&id, "note", 0.5));
            g.add_edge(&id, "hub", "LINKED_TO", 1.0 - i as f64 / 1000.0);
        }
        // the hub reached at hop 2 (from a note) is capped at `fanout`
        g.add_node(node("start", "note", 0.5));
        g.add_edge("start", "hub", "LINKED_TO", 1.0);
        let mut p = params(2, 0.0, 400);
        p.fanout = 10;
        let raw = expand_in_memory(&g, "note", "start", &p).unwrap();
        assert_eq!(raw.nodes.len(), 1 + 10);
        // strongest kept
        assert!(raw.nodes.iter().any(|n| n.id == "n000"));
        assert!(!raw.nodes.iter().any(|n| n.id == "n099"));
        // as the centre, the hub may contribute up to `limit` neighbours
        let mut p = params(1, 0.0, 50);
        p.fanout = 10;
        let raw = expand_in_memory(&g, "file", "hub", &p).unwrap();
        assert_eq!(raw.nodes.len(), 50);
    }

    #[test]
    fn selection_is_deterministic() {
        let g = sample();
        let p = params(3, 0.0, 4);
        let a = run(&g, ("note", "n0"), &p);
        let b = run(&g, ("note", "n0"), &p);
        assert_eq!(a, b);
    }

    #[test]
    fn unknown_center_is_none() {
        let g = sample();
        assert!(expand_in_memory(&g, "note", "nope", &params(1, 0.0, 10)).is_none());
        // right id, wrong type
        assert!(expand_in_memory(&g, "task", "n0", &params(1, 0.0, 10)).is_none());
    }

    #[test]
    fn params_are_clamped() {
        let p = NeighborhoodParams::clamped(Some(9), Some(4.0), Some(10_000), vec![]);
        assert_eq!((p.depth, p.min_weight, p.limit), (3, 1.0, 400));
        let p = NeighborhoodParams::clamped(Some(0), Some(-1.0), Some(0), vec![]);
        assert_eq!((p.depth, p.min_weight, p.limit), (1, 0.0, 1));
        let p = NeighborhoodParams::clamped(None, None, None, vec![]);
        assert_eq!((p.depth, p.min_weight, p.limit), (2, 0.0, 120));
    }

    #[test]
    fn salience_per_type() {
        let p = NodeProps {
            energy: Some(0.42),
            ..Default::default()
        };
        assert!((node_salience("note", &p) - 0.42).abs() < 1e-9);
        let p = NodeProps {
            pagerank: Some(1.0),
            ..Default::default()
        };
        assert!((node_salience("file", &p) - 1.0).abs() < 1e-9);
        assert!((pagerank_salience(1e-3) - 0.4).abs() < 1e-9);
        assert_eq!(pagerank_salience(0.0), 0.0);
        let p = NodeProps {
            status: Some("InProgress".into()),
            priority: Some(100.0),
            ..Default::default()
        };
        assert!((node_salience("task", &p) - 1.0).abs() < 1e-9);
        let p = NodeProps {
            message_count: Some(50.0),
            ..Default::default()
        };
        assert!((node_salience("chat_session", &p) - 0.5).abs() < 1e-9);
        let p = NodeProps {
            energy: Some(7.0),
            ..Default::default()
        };
        assert_eq!(node_salience("note", &p), 1.0);
    }

    #[test]
    fn labels_are_short_and_human() {
        let p = NodeProps {
            content: Some("\n## A heading\nbody".into()),
            note_type: Some("gotcha".into()),
            importance: Some("high".into()),
            status: Some("active".into()),
            ..Default::default()
        };
        let (l, s) = node_label("note", "id", &p);
        assert_eq!(l, "A heading");
        assert_eq!(s.as_deref(), Some("gotcha · high · active"));

        let p = NodeProps {
            path: Some("/a/b/src/lib.rs".into()),
            ..Default::default()
        };
        let (l, s) = node_label("file", "/a/b/src/lib.rs", &p);
        assert_eq!(l, "lib.rs");
        assert_eq!(s.as_deref(), Some("/a/b/src/lib.rs"));

        let p = NodeProps {
            content: Some("x".repeat(200)),
            ..Default::default()
        };
        let (l, _) = node_label("note", "id", &p);
        assert_eq!(l.chars().count(), 81);

        let (l, s) = node_label("decision", "the-id", &NodeProps::default());
        assert_eq!(l, "the-id");
        assert_eq!(s, None);
    }

    #[test]
    fn kinds_lookup() {
        assert_eq!(entity_kind("feature_graph").unwrap().label, "FeatureGraph");
        assert_eq!(entity_kind("file").unwrap().id_prop, "path");
        assert_eq!(entity_kind("commit").unwrap().id_prop, "hash");
        assert!(entity_kind("import").is_none());
        assert_eq!(kind_for_labels(&["Document"]).unwrap().api, "document");
        assert!(kind_for_labels(&["ChatEvent"]).is_none());
    }
}
