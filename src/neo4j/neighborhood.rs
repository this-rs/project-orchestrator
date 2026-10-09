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
    all_labels, entity_kind, kind_for_labels, layer_of, next_frontier, rel_types_for,
    NeighborhoodParams, NodeProps, ProjectFilter, RawEdge, RawNeighborhood, RawNode,
    ScopedNeighborhood, ScopedNode, EDGE_WEIGHT_CYPHER,
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

/// Owning project id of node `m`, as the project filter reads it. A node carries
/// `project_id` itself, is a Project, or hangs under a Plan of a Project (Plan,
/// Task, Step). Anything else has no known owner (`null`): the filter refuses it.
/// NOT run against a real database (written from the schema, see the PR).
const OWNER_CYPHER: &str = "coalesce(m.project_id, \
    CASE WHEN m:Project THEN m.id END, \
    head([(op:Project)-[:HAS_PLAN]->(m) | op.id]), \
    head([(op:Project)-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(m) | op.id]), \
    head([(op:Project)-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(:Task)-[:HAS_STEP]->(m) | op.id]))";

impl Neo4jClient {
    /// Bounded candidate neighbourhood around `(center_type, center_id)`.
    /// Returns `Ok(None)` when the centre does not exist.
    pub async fn get_entity_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &NeighborhoodParams,
    ) -> Result<Option<RawNeighborhood>> {
        Ok(self
            .fetch_neighborhood(center_type, center_id, params, None)
            .await?
            .map(|s| RawNeighborhood {
                center: s.center.map(|c| c.node),
                nodes: s.nodes.into_iter().map(|n| n.node).collect(),
                edges: s.edges,
            }))
    }

    /// Same walk, with the PROJECT filter applied inside the hop query, before the
    /// per-node `LIMIT`: a neighbour owned by another project (or by nobody known)
    /// can never evict a local one. The centre is always returned, with its owner,
    /// so the caller can run the consent predicate on it.
    pub async fn get_scoped_entity_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &NeighborhoodParams,
        filter: &ProjectFilter,
    ) -> Result<Option<ScopedNeighborhood>> {
        self.fetch_neighborhood(center_type, center_id, params, Some(filter))
            .await
    }

    async fn fetch_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &NeighborhoodParams,
        filter: Option<&ProjectFilter>,
    ) -> Result<Option<ScopedNeighborhood>> {
        let kind = entity_kind(center_type)
            .ok_or_else(|| anyhow!("unsupported entity type '{}'", center_type))?;

        // --- centre ---
        let center_q = format!(
            "MATCH (m:{label} {{{prop}: $id}}) \
             RETURN elementId(m) AS eid, m.{prop} AS pid, {props} AS props, \
             {owner} AS owner, toString(m.sharing_consent) AS consent LIMIT 1",
            owner = OWNER_CYPHER,
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
        let center = ScopedNode {
            node: RawNode::from_props(kind.api, center_pid.clone(), &props),
            project_id: row.get::<String>("owner").ok(),
            consent: parse_consent(row.get::<String>("consent").ok().as_deref()),
        };

        let rels = rel_types_for(&params.layers);
        if rels.is_empty() {
            return Ok(Some(ScopedNeighborhood {
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
               WHERE ({label_filter}){project_filter} \
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
                    coalesce(m.id, m.path, m.hash) AS pid, {props} AS props, \
                    {owner_ret} AS owner, toString(m.sharing_consent) AS consent",
            owner_ret = if filter.is_some() {
                OWNER_CYPHER
            } else {
                "null"
            },
            project_filter = if filter.is_some() {
                format!(" AND {OWNER_CYPHER} IN $projects")
            } else {
                String::new()
            },
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
        let projects: Vec<String> = filter.map(|f| f.project_ids.clone()).unwrap_or_default();
        let mut nodes: Vec<ScopedNode> = Vec::new();
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

            let mut q = query(&hop_q)
                .param("frontier", expandable)
                .param("min_weight", params.min_weight)
                .param("fanout", params.fanout_for_hop(hop) as i64);
            if filter.is_some() {
                q = q.param("projects", projects.clone());
            }
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
                    let node = ScopedNode {
                        node,
                        project_id: row.get::<String>("owner").ok(),
                        consent: parse_consent(row.get::<String>("consent").ok().as_deref()),
                    };
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

            // Rank by PUBLIC id, like the in-memory walk: an elementId is an internal
            // counter, so breaking ties on it made the walk depend on creation order.
            let by_pid: HashMap<String, (f64, f64)> = best
                .iter()
                .map(|(eid, v)| (known[eid].0.clone(), *v))
                .collect();
            let eid_of: HashMap<&str, &String> = best
                .keys()
                .map(|eid| (known[eid].0.as_str(), eid))
                .collect();
            frontier = next_frontier(&by_pid, params.frontier_cap)
                .iter()
                .map(|pid| eid_of[pid.as_str()].clone())
                .collect();
            if frontier.is_empty() {
                break;
            }
        }

        Ok(Some(ScopedNeighborhood {
            center: Some(center),
            nodes,
            edges,
        }))
    }
}

/// Stored `sharing_consent` -> enum. Unknown or absent is `NotSet` (never permissive).
fn parse_consent(s: Option<&str>) -> crate::episodes::distill_models::SharingConsent {
    s.and_then(|s| serde_json::from_value(serde_json::Value::String(s.to_string())).ok())
        .unwrap_or_default()
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
