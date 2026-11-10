//! Le catalogue du schéma et la marche causale, côté Neo4j.
//!
//! Deux requêtes de natures très différentes vivent ici, et c'est délibéré :
//!
//! * [`Neo4jClient::schema_catalog`] lit la **forme** du graphe par `apoc.meta.schema`,
//!   puis les **volumes** par le count-store, un appel par type. Les volumes d'apoc sont
//!   jetés : mesuré le 30/10, il annonçait 0 `INFORMED_BY` là où il y en a 833.
//! * [`Neo4jClient::causal_walk`] part d'un nœud ancré par `(label, id)` — donc par index
//!   — et n'étend que les types de la famille causale, écrits dans le motif de longueur
//!   variable. Une expansion non filtrée sur 3,3 M de nœuds ne rendrait jamais la main.

use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

use crate::graph::schema::{
    CausalStep, CausalWalk, PathNode, RelationShape, SchemaCatalog, TypedPath, CAUSAL_ANCHORS,
    CAUSAL_TYPES,
};
use crate::neo4j::client::Neo4jClient;

impl Neo4jClient {
    /// See [`crate::neo4j::AnalyticsStore::schema_catalog`].
    pub async fn schema_catalog(&self) -> Result<SchemaCatalog> {
        let mut labels = Vec::new();
        let mut rows = self
            .graph
            .execute(query(
                "CALL db.labels() YIELD label RETURN label ORDER BY label",
            ))
            .await?;
        while let Some(row) = rows.next().await? {
            labels.push(row.get::<String>("label")?);
        }

        let mut types = Vec::new();
        let mut rows = self
            .graph
            .execute(query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType \
                 ORDER BY relationshipType",
            ))
            .await?;
        while let Some(row) = rows.next().await? {
            types.push(row.get::<String>("relationshipType")?);
        }

        // The shape: which labels each type leaves and reaches. `direction: 'out'` only,
        // so every pair is described once, from its source.
        let mut edges = Vec::new();
        let mut rows = self
            .graph
            .execute(query(
                "CALL apoc.meta.schema() YIELD value \
                 UNWIND keys(value) AS lab \
                 WITH lab, value[lab] AS node \
                 WHERE node.type = 'node' AND node.relationships IS NOT NULL \
                 UNWIND keys(node.relationships) AS edge \
                 WITH lab, edge, node.relationships[edge] AS info \
                 WHERE info.direction = 'out' \
                 UNWIND coalesce(info.labels, ['']) AS other \
                 RETURN DISTINCT edge AS edge, lab AS f, other AS t \
                 ORDER BY edge, f, t",
            ))
            .await?;
        while let Some(row) = rows.next().await? {
            edges.push(RelationShape {
                edge: row.get("edge")?,
                from: row.get("f")?,
                to: row.get("t")?,
            });
        }

        // The volume, per type, from the count store: exact, and cheap enough to ask of
        // every type the graph declares.
        let mut volumes = std::collections::BTreeMap::new();
        for t in &types {
            if !t.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()) {
                // A type that is not a bare identifier is not interpolated into a query.
                continue;
            }
            let mut rows = self
                .graph
                .execute(query(&format!("MATCH ()-[r:{t}]->() RETURN count(r) AS n")))
                .await?;
            if let Some(row) = rows.next().await? {
                let n: i64 = row.get("n").unwrap_or(0);
                volumes.insert(t.clone(), u64::try_from(n).unwrap_or(0));
            }
        }

        Ok(SchemaCatalog {
            labels,
            types,
            edges,
            volumes,
        })
    }

    /// See [`crate::neo4j::AnalyticsStore::causal_walk`].
    pub async fn causal_walk(&self, label: &str, id: &str, max_depth: u32) -> Result<CausalWalk> {
        if !CAUSAL_ANCHORS.contains(&label) {
            anyhow::bail!(
                "{label} is not a causal anchor: {}",
                CAUSAL_ANCHORS.join(", ")
            );
        }
        // Clamped, and an integer: the depth is interpolated into the pattern, never a
        // caller-supplied string.
        let depth = max_depth.clamp(1, 4);
        let family = CAUSAL_TYPES.join("|");
        let statement = format!(
            "MATCH (a:{label} {{id: $id}}) \
             MATCH p = (a)-[:{family}*1..{depth}]->(b) \
             UNWIND range(0, size(relationships(p)) - 1) AS i \
             WITH i, relationships(p)[i] AS r, nodes(p)[i + 1] AS n \
             RETURN i + 1 AS depth, type(r) AS edge, labels(n)[0] AS label, \
                    n.id AS id, coalesce(n.title, n.name, n.slug, n.path) AS title \
             ORDER BY depth, n.id ASC LIMIT 200"
        );
        let mut rows = self
            .graph
            .execute(query(&statement).param("id", id.to_string()))
            .await?;

        let mut steps = Vec::new();
        let mut seen: std::collections::HashSet<(u32, String, String)> = Default::default();
        while let Some(row) = rows.next().await? {
            let depth: i64 = row.get("depth").unwrap_or(0);
            let edge: String = row.get("edge")?;
            let label: String = row.get("label")?;
            let node_id: String = row.get("id").unwrap_or_default();
            if !seen.insert((
                u32::try_from(depth).unwrap_or(0),
                edge.clone(),
                node_id.clone(),
            )) {
                continue;
            }
            steps.push(CausalStep {
                depth: u32::try_from(depth).unwrap_or(0),
                edge,
                label,
                id: node_id,
                title: row.get::<Option<String>>("title").unwrap_or(None),
            });
        }

        // Truncation is measured, not assumed: a step sitting on the depth limit may still
        // have edges to follow, and saying so costs one read.
        let at_limit: Vec<&CausalStep> = steps.iter().filter(|s| s.depth >= depth).collect();
        let mut truncated = false;
        for step in at_limit {
            if !step
                .label
                .chars()
                .all(|c| c == '_' || c.is_ascii_alphanumeric())
            {
                continue;
            }
            let mut rows = self
                .graph
                .execute(
                    query(&format!(
                        "MATCH (n:{0} {{id: $id}}) MATCH (n)-[r:{1}]->() RETURN count(r) AS n",
                        step.label,
                        CAUSAL_TYPES.join("|")
                    ))
                    .param("id", step.id.clone()),
                )
                .await?;
            if let Some(row) = rows.next().await? {
                if row.get::<i64>("n").unwrap_or(0) > 0 {
                    truncated = true;
                    break;
                }
            }
        }

        Ok(CausalWalk {
            anchor_label: label.to_string(),
            anchor_id: id.to_string(),
            max_depth: depth,
            edges_followed: CAUSAL_TYPES.iter().map(|s| (*s).to_string()).collect(),
            steps,
            truncated,
        })
    }

    /// See [`crate::neo4j::AnalyticsStore::typed_path`]. Two reads, both read-only: the
    /// existence of each endpoint (exact, by index), then the shortest causal chain between
    /// them, only when both exist.
    pub async fn typed_path(
        &self,
        from_label: &str,
        from_id: &str,
        to_label: &str,
        to_id: &str,
        max_depth: u32,
    ) -> Result<TypedPath> {
        for label in [from_label, to_label] {
            if !CAUSAL_ANCHORS.contains(&label) {
                anyhow::bail!(
                    "{label} is not a causal anchor: {}",
                    CAUSAL_ANCHORS.join(", ")
                );
            }
        }
        if from_label == to_label && from_id == to_id {
            anyhow::bail!("the path needs two distinct nodes, got {from_label} {from_id} twice");
        }
        // Clamped, and an integer: the depth is interpolated, never a caller string.
        let depth = max_depth.clamp(1, 4);
        let family = CAUSAL_TYPES.join("|");
        let node = |label: &str, id: &str| PathNode {
            label: label.to_string(),
            id: id.to_string(),
            title: None,
        };
        let mut found = self
            .graph
            .execute(
                query(&format!(
                    "MATCH (a:{from_label} {{id: $src}}) MATCH (b:{to_label} {{id: $dst}}) \
                     RETURN count(*) AS n"
                ))
                .param("src", from_id.to_string())
                .param("dst", to_id.to_string()),
            )
            .await?;
        let both_exist = match found.next().await? {
            Some(row) => row.get::<i64>("n").unwrap_or(0) > 0,
            None => false,
        };
        let mut result = TypedPath {
            from: node(from_label, from_id),
            to: node(to_label, to_id),
            max_depth: depth,
            edges_followed: CAUSAL_TYPES.iter().map(|s| (*s).to_string()).collect(),
            endpoints_found: both_exist,
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        if !both_exist {
            return Ok(result);
        }

        let mut rows = self
            .graph
            .execute(
                query(&format!(
                    "MATCH (a:{from_label} {{id: $src}}), (b:{to_label} {{id: $dst}}) \
                     MATCH p = shortestPath((a)-[:{family}*1..{depth}]->(b)) \
                     RETURN [n IN nodes(p) | labels(n)[0]] AS labels, \
                            [n IN nodes(p) | n.id] AS ids, \
                            [n IN nodes(p) | coalesce(n.title, n.name, n.slug, n.path)] AS titles, \
                            [r IN relationships(p) | type(r)] AS edges"
                ))
                .param("src", from_id.to_string())
                .param("dst", to_id.to_string()),
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let labels: Vec<String> = row.get("labels")?;
            let ids: Vec<String> = row.get("ids")?;
            let titles: Vec<Option<String>> = row.get("titles").unwrap_or_default();
            result.edges = row.get("edges")?;
            result.nodes = labels
                .into_iter()
                .zip(ids)
                .enumerate()
                .map(|(i, (label, id))| PathNode {
                    label,
                    id,
                    title: titles.get(i).cloned().flatten(),
                })
                .collect();
        }
        Ok(result)
    }

    /// The agent node of an id, when it exists — the read `assign_task` has always had
    /// without a writer, now that [`Neo4jClient::ensure_agent`] writes it.
    pub async fn find_agent(&self, agent_id: Uuid) -> Result<Option<(String, Option<String>)>> {
        let mut rows = self
            .graph
            .execute(
                query("MATCH (a:Agent {id: $id}) RETURN a.kind AS kind, a.model AS model")
                    .param("id", agent_id.to_string()),
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some((
                row.get::<String>("kind").unwrap_or_default(),
                row.get::<Option<String>>("model").unwrap_or(None),
            ))),
            None => Ok(None),
        }
    }
}
