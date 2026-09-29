//! Neo4j fetch for the entity neighbourhood (see `crate::graph::neighborhood`).
//!
//! One query resolves the centre, then one query per hop expands the current
//! frontier. Each hop query is bounded: every frontier node contributes at
//! most `fanout` relationships (strongest first, sorted inside a per-node
//! subquery), the frontier itself is capped at `frontier_cap` nodes, and the
//! traversal never uses variable-length patterns. Nodes are carried between
//! hops by `elementId`, which Neo4j resolves with a direct seek.

use super::client::Neo4jClient;
use crate::graph::neighborhood::{
    all_labels, entity_kind, kind_for_labels, layer_of, next_frontier, rel_types_for, NodeProps,
    NeighborhoodParams, RawEdge, RawNeighborhood, RawNode, EDGE_WEIGHT_CYPHER,
};
use anyhow::{anyhow, Result};
use neo4rs::query;
use std::collections::{HashMap, HashSet};

/// Map projection of the properties read by `NodeProps` (`m` = node).
/// Long texts are cut server-side so a hop never ships whole note bodies.
const PROPS_PROJECTION: &str = "{\
    id: m.id, name: m.name, title: m.title, slug: m.slug, path: m.path, hash: m.hash, \
    filename: m.filename, format: m.format, \
    content: left(m.content, 200), description: left(m.description, 200), \
    message: left(m.message, 200), preview: left(m.preview, 120), \
    note_type: toString(m.note_type), importance: toString(m.importance), \
    status: toString(m.status), priority: toFloat(m.priority), energy: toFloat(m.energy), \
    pagerank: toFloat(m.pagerank), message_count: toFloat(m.message_count), \
    model: m.model, file_path: m.file_path, line_start: toInteger(m.line_start), \
    state_type: toString(m.state_type), constraint_type: toString(m.constraint_type), \
    protocol_category: toString(m.protocol_category), author: m.author}";

impl Neo4jClient {
    /// Bounded candidate neighbourhood around `(center_type, center_id)`.
    /// Returns `Ok(None)` when the centre does not exist.
    pub async fn get_entity_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &NeighborhoodParams,
    ) -> Result<Option<RawNeighborhood>> {
        let kind = entity_kind(center_type)
            .ok_or_else(|| anyhow!("unsupported entity type '{}'", center_type))?;

        // --- centre ---
        let center_q = format!(
            "MATCH (m:{label} {{{prop}: $id}}) \
             RETURN elementId(m) AS eid, m.{prop} AS pid, {props} AS props LIMIT 1",
            label = kind.label,
            prop = kind.id_prop,
            props = PROPS_PROJECTION,
        );
        let mut res = self
            .graph
            .execute(query(&center_q).param("id", center_id.to_string()))
            .await?;
        let Some(row) = res.next().await? else {
            return Ok(None);
        };
        let center_eid: String = row.get("eid")?;
        let center_pid: String = row.get("pid")?;
        let props: NodeProps = row.get("props").unwrap_or_default();
        let center = RawNode::from_props(kind.api, center_pid.clone(), &props);

        let rels = rel_types_for(&params.layers);
        if rels.is_empty() {
            return Ok(Some(RawNeighborhood {
                center: Some(center),
                ..Default::default()
            }));
        }

        // Relationship types and labels come from compile-time whitelists,
        // so interpolating them is safe (and lets the planner use the
        // per-type relationship chains instead of filtering every edge).
        let hop_q = format!(
            "UNWIND $frontier AS fid \
             MATCH (n) WHERE elementId(n) = fid \
             CALL {{ \
               WITH n \
               MATCH (n)-[r:{rels}]-(m) \
               WHERE {label_filter} \
               WITH r, m, {weight} AS w \
               WHERE w >= $min_weight \
               WITH r, m, w \
               ORDER BY w DESC, coalesce(m.energy, m.pagerank, 0.0) DESC, \
                        coalesce(m.id, m.path, m.hash) ASC \
               LIMIT $fanout \
               RETURN r, m, w \
             }} \
             RETURN fid, elementId(m) AS mid, elementId(startNode(r)) = fid AS outgoing, \
                    type(r) AS rel, w, labels(m) AS labels, \
                    coalesce(m.id, m.path, m.hash) AS pid, {props} AS props",
            rels = rels.join("|"),
            label_filter = all_labels()
                .iter()
                .map(|l| format!("m:{l}"))
                .collect::<Vec<_>>()
                .join(" OR "),
            weight = EDGE_WEIGHT_CYPHER,
            props = PROPS_PROJECTION,
        );

        // elementId → (public id, api type)
        let mut known: HashMap<String, (String, &'static str)> = HashMap::new();
        known.insert(center_eid.clone(), (center_pid.clone(), kind.api));
        let mut nodes: Vec<RawNode> = Vec::new();
        let mut edges: Vec<RawEdge> = Vec::new();
        let mut edge_keys: HashSet<(String, String, String)> = HashSet::new();
        let mut frontier: Vec<String> = vec![center_eid.clone()];

        for hop in 1..=params.depth {
            // Containers are leaves unless they are the centre.
            let expandable: Vec<String> = frontier
                .iter()
                .filter(|eid| {
                    **eid == center_eid
                        || known
                            .get(*eid)
                            .and_then(|(_, t)| entity_kind(t))
                            .is_none_or(|k| !k.container)
                })
                .cloned()
                .collect();
            if expandable.is_empty() {
                break;
            }

            let q = query(&hop_q)
                .param("frontier", expandable)
                .param("min_weight", params.min_weight)
                .param("fanout", params.fanout_for_hop(hop) as i64);
            let mut res = self.graph.execute(q).await?;

            // new node elementId → (best incoming weight, salience)
            let mut best: HashMap<String, (f64, f64)> = HashMap::new();
            while let Some(row) = res.next().await? {
                let fid: String = row.get("fid")?;
                let mid: String = row.get("mid")?;
                let outgoing: bool = row.get("outgoing").unwrap_or(true);
                let rel: String = row.get("rel")?;
                let w: f64 = row.get::<f64>("w").unwrap_or(1.0).clamp(0.0, 1.0);
                let labels: Vec<String> = row.get("labels").unwrap_or_default();
                let Some(mkind) = kind_for_labels(&labels) else {
                    continue;
                };
                let Ok(pid) = row.get::<String>("pid") else {
                    continue;
                };
                let Some((fpid, ftype)) = known.get(&fid).cloned() else {
                    continue;
                };

                let (src_type, src, tgt) = if outgoing {
                    (ftype, fpid, pid.clone())
                } else {
                    (mkind.api, pid.clone(), fpid)
                };
                let Some(layer) = layer_of(&rel, src_type) else {
                    continue;
                };
                if !params.layers.contains(&layer) {
                    continue;
                }

                if !known.contains_key(&mid) {
                    let props: NodeProps = row.get("props").unwrap_or_default();
                    let node = RawNode::from_props(mkind.api, pid.clone(), &props);
                    best.insert(mid.clone(), (w, node.weight));
                    known.insert(mid.clone(), (pid.clone(), mkind.api));
                    nodes.push(node);
                } else if let Some(entry) = best.get_mut(&mid) {
                    entry.0 = entry.0.max(w);
                }

                if edge_keys.insert((src.clone(), tgt.clone(), rel.clone())) {
                    edges.push(RawEdge {
                        source: src,
                        target: tgt,
                        rel,
                        weight: w,
                        layer,
                    });
                }
            }

            frontier = next_frontier(&best, params.frontier_cap);
            if frontier.is_empty() {
                break;
            }
        }

        Ok(Some(RawNeighborhood {
            center: Some(center),
            nodes,
            edges,
        }))
    }
}

#[cfg(test)]
mod bench {
    //! Timing against a real graph. Read-only (no schema init, MATCH only).
    //! `NEIGHBORHOOD_BENCH=bolt://host:7687,user,password \
    //!  NEIGHBORHOOD_BENCH_CENTERS='note:<id>;file:<path>' \
    //!  cargo test --lib neighborhood_bench -- --ignored --nocapture`
    use super::*;
    use crate::graph::neighborhood::{select_neighborhood, Layer};

    #[tokio::test]
    #[ignore]
    async fn neighborhood_bench() {
        let Ok(conn) = std::env::var("NEIGHBORHOOD_BENCH") else {
            return;
        };
        let parts: Vec<&str> = conn.splitn(3, ',').collect();
        let config = neo4rs::ConfigBuilder::new()
            .uri(parts[0])
            .user(parts[1])
            .password(parts[2])
            .build()
            .unwrap();
        let client = Neo4jClient {
            graph: std::sync::Arc::new(neo4rs::Graph::connect(config).await.unwrap()),
            coupling_cache: Default::default(),
        };
        let centers = std::env::var("NEIGHBORHOOD_BENCH_CENTERS").unwrap_or_default();
        for c in centers.split(';').filter(|c| !c.is_empty()) {
            let (t, id) = c.split_once(':').unwrap();
            for depth in 1..=3 {
                let p = NeighborhoodParams::clamped(Some(depth), None, None, Layer::ALL.to_vec());
                let mut times = Vec::new();
                let mut summary = String::new();
                for _ in 0..3 {
                    let t0 = std::time::Instant::now();
                    let raw = client
                        .get_entity_neighborhood(t, id, &p)
                        .await
                        .unwrap()
                        .expect("center exists");
                    let r = select_neighborhood(&raw, &p).unwrap();
                    times.push(t0.elapsed().as_millis());
                    let json = serde_json::to_string(&r).unwrap();
                    summary = format!(
                        "raw {}n/{}e -> {}n/{}e, total_before_limit {}, truncated {}, {} bytes",
                        raw.nodes.len(),
                        raw.edges.len(),
                        r.nodes.len(),
                        r.edges.len(),
                        r.stats.total_before_limit,
                        r.truncated,
                        json.len()
                    );
                    if depth == 2 && t == "note" {
                        std::fs::write("/tmp/neighborhood_note_d2.json", &json).ok();
                    }
                }
                println!("BENCH {t} depth={depth} ms={times:?} {summary}");
            }
        }
    }
}
