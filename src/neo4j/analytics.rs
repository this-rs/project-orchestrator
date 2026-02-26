//! Neo4j Analytics operations (communities, health, GDS metrics)

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    /// Get distinct communities for a project (from graph analytics Louvain clustering).
    /// Returns communities sorted by file_count descending.
    pub async fn get_project_communities(&self, project_id: Uuid) -> Result<Vec<CommunityRow>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.community_id IS NOT NULL
            WITH f.community_id AS cid, f.community_label AS label,
                 count(f) AS file_count,
                 collect(f.path) AS all_paths
            ORDER BY file_count DESC
            RETURN cid, label, file_count,
                   [p IN all_paths | p][..3] AS key_files
            "#,
        )
        .param("pid", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut communities = Vec::new();

        while let Some(row) = result.next().await? {
            let community_id = row.get::<i64>("cid").unwrap_or(0);
            let community_label = row
                .get::<String>("label")
                .unwrap_or_else(|_| format!("Community {}", community_id));
            let file_count = row.get::<i64>("file_count").unwrap_or(0) as usize;
            let key_files: Vec<String> = row.get::<Vec<String>>("key_files").unwrap_or_default();

            communities.push(CommunityRow {
                community_id,
                community_label,
                file_count,
                key_files,
            });
        }

        Ok(communities)
    }

    /// Get GDS analytics properties for a node (File by path, or Function by name).
    pub async fn get_node_analytics(
        &self,
        identifier: &str,
        node_type: &str,
    ) -> Result<Option<NodeAnalyticsRow>> {
        let cypher = if node_type == "function" {
            r#"
            MATCH (n:Function {name: $id})
            RETURN n.pagerank AS pagerank, n.betweenness AS betweenness,
                   n.community_id AS community_id, n.community_label AS community_label
            LIMIT 1
            "#
        } else {
            r#"
            MATCH (n:File {path: $id})
            RETURN n.pagerank AS pagerank, n.betweenness AS betweenness,
                   n.community_id AS community_id, n.community_label AS community_label
            LIMIT 1
            "#
        };

        let q = query(cypher).param("id", identifier);
        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            Ok(Some(NodeAnalyticsRow {
                pagerank: row.get::<f64>("pagerank").ok(),
                betweenness: row.get::<f64>("betweenness").ok(),
                community_id: row.get::<i64>("community_id").ok(),
                community_label: row.get::<String>("community_label").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get distinct community labels for a list of file paths.
    pub async fn get_affected_communities(&self, file_paths: &[String]) -> Result<Vec<String>> {
        if file_paths.is_empty() {
            return Ok(vec![]);
        }

        let q = query(
            r#"
            MATCH (f:File)
            WHERE f.path IN $paths AND f.community_label IS NOT NULL
            RETURN DISTINCT f.community_label AS label
            ORDER BY label
            "#,
        )
        .param("paths", file_paths.to_vec());

        let mut result = self.graph.execute(q).await?;
        let mut labels = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(label) = row.get::<String>("label") {
                labels.push(label);
            }
        }

        Ok(labels)
    }

    /// Get a structural health report for a project: god functions, orphan files, coupling metrics.
    pub async fn get_code_health_report(
        &self,
        project_id: Uuid,
        god_function_threshold: usize,
    ) -> Result<CodeHealthReport> {
        use crate::neo4j::models::{CodeHealthReport, CouplingMetrics, GodFunction};

        // God functions: functions with high in-degree (many callers)
        let god_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)-[:CONTAINS]->(func:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(func)
            OPTIONAL MATCH (func)-[:CALLS]->(callee:Function)
            WITH func, f, count(DISTINCT caller) AS in_deg, count(DISTINCT callee) AS out_deg
            WHERE in_deg >= $threshold
            RETURN func.name AS name, f.path AS file, in_deg, out_deg
            ORDER BY in_deg DESC
            LIMIT 10
            "#,
        )
        .param("pid", project_id.to_string())
        .param("threshold", god_function_threshold as i64);

        let god_rows = self.execute_with_params(god_q).await?;
        let god_functions: Vec<GodFunction> = god_rows
            .iter()
            .filter_map(|row| {
                let name = row.get::<String>("name").ok()?;
                let file = row.get::<String>("file").ok()?;
                let in_degree = row.get::<i64>("in_deg").unwrap_or(0) as usize;
                let out_degree = row.get::<i64>("out_deg").unwrap_or(0) as usize;
                Some(GodFunction {
                    name,
                    file,
                    in_degree,
                    out_degree,
                })
            })
            .collect();

        // Orphan files: files with no IMPORTS relationships (neither importing nor imported)
        let orphan_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE NOT EXISTS { (f)-[:IMPORTS]->() }
              AND NOT EXISTS { ()-[:IMPORTS]->(f) }
              AND NOT EXISTS { (f)-[:CONTAINS]->(:Function) }
            RETURN f.path AS path
            ORDER BY path
            LIMIT 20
            "#,
        )
        .param("pid", project_id.to_string());

        let orphan_rows = self.execute_with_params(orphan_q).await?;
        let orphan_files: Vec<String> = orphan_rows
            .iter()
            .filter_map(|row| row.get::<String>("path").ok())
            .collect();

        // Coupling metrics from clustering_coefficient
        let coupling_q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.clustering_coefficient IS NOT NULL
            WITH avg(f.clustering_coefficient) AS avg_cc,
                 max(f.clustering_coefficient) AS max_cc,
                 collect({path: f.path, cc: f.clustering_coefficient}) AS files
            WITH avg_cc, max_cc, files,
                 [x IN files WHERE x.cc = max_cc | x.path][0] AS most_coupled
            RETURN avg_cc, max_cc, most_coupled
            "#,
        )
        .param("pid", project_id.to_string());

        let coupling_rows = self.execute_with_params(coupling_q).await?;
        let coupling_metrics = coupling_rows.first().and_then(|row| {
            let avg = row.get::<f64>("avg_cc").ok()?;
            let max = row.get::<f64>("max_cc").ok()?;
            let most_coupled = row.get::<String>("most_coupled").ok();
            Some(CouplingMetrics {
                avg_clustering_coefficient: avg,
                max_clustering_coefficient: max,
                most_coupled_file: most_coupled,
            })
        });

        Ok(CodeHealthReport {
            god_functions,
            orphan_files,
            coupling_metrics,
        })
    }

    /// Detect circular dependencies between files (import cycles).
    pub async fn get_circular_dependencies(&self, project_id: Uuid) -> Result<Vec<Vec<String>>> {
        let q = query(
            r#"
            MATCH path = (f:File)-[:IMPORTS*2..5]->(f)
            WHERE EXISTS { MATCH (p:Project {id: $pid})-[:CONTAINS]->(f) }
            WITH nodes(path) AS cycle_nodes
            WITH [n IN cycle_nodes | n.path] AS cycle
            WITH cycle, cycle[0] AS canonical
            RETURN DISTINCT cycle
            ORDER BY size(cycle)
            LIMIT 10
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        let mut cycles: Vec<Vec<String>> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(cycle) = row.get::<Vec<String>>("cycle") {
                // Deduplicate: normalize by sorting the cycle to a canonical form
                let mut canonical = cycle.clone();
                canonical.sort();
                let key = canonical.join("|");
                if seen.insert(key) {
                    cycles.push(cycle);
                }
            }
        }

        Ok(cycles)
    }

    /// Get GDS metrics for a specific node (file or function) in a project.
    pub async fn get_node_gds_metrics(
        &self,
        node_path: &str,
        node_type: &str,
        project_id: Uuid,
    ) -> Result<Option<NodeGdsMetrics>> {
        let (id_prop, match_pattern) = match node_type {
            "function" => (
                "name",
                "MATCH (p:Project {id: $pid})-[:CONTAINS]->(:File)-[:CONTAINS]->(n:Function {name: $node_path})",
            ),
            _ => (
                "path",
                "MATCH (p:Project {id: $pid})-[:CONTAINS]->(n:File {path: $node_path})",
            ),
        };

        let cypher = format!(
            r#"
            {match_pattern}
            OPTIONAL MATCH (caller)-[]->(n)
            WITH n, count(DISTINCT caller) AS in_deg
            OPTIONAL MATCH (n)-[]->(callee)
            WITH n, in_deg, count(DISTINCT callee) AS out_deg
            RETURN
                n.{id_prop} AS node_path,
                n.pagerank AS pagerank,
                n.betweenness AS betweenness,
                n.clustering_coefficient AS clustering_coefficient,
                n.community_id AS community_id,
                n.community_label AS community_label,
                n.fabric_pagerank AS fabric_pagerank,
                n.fabric_betweenness AS fabric_betweenness,
                n.fabric_community_id AS fabric_community_id,
                n.fabric_community_label AS fabric_community_label,
                in_deg, out_deg
            "#,
        );

        let q = query(&cypher)
            .param("pid", project_id.to_string())
            .param("node_path", node_path);

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let path: String = row.get("node_path").unwrap_or_default();
            if path.is_empty() {
                return Ok(None);
            }
            Ok(Some(NodeGdsMetrics {
                node_path: path,
                node_type: node_type.to_string(),
                pagerank: row.get::<f64>("pagerank").ok(),
                betweenness: row.get::<f64>("betweenness").ok(),
                clustering_coefficient: row.get::<f64>("clustering_coefficient").ok(),
                community_id: row.get::<i64>("community_id").ok(),
                community_label: row.get::<String>("community_label").ok(),
                in_degree: row.get::<i64>("in_deg").unwrap_or(0),
                out_degree: row.get::<i64>("out_deg").unwrap_or(0),
                fabric_pagerank: row.get::<f64>("fabric_pagerank").ok(),
                fabric_betweenness: row.get::<f64>("fabric_betweenness").ok(),
                fabric_community_id: row.get::<i64>("fabric_community_id").ok(),
                fabric_community_label: row.get::<String>("fabric_community_label").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get statistical percentiles for GDS metrics across all files+functions in a project.
    pub async fn get_project_percentiles(&self, project_id: Uuid) -> Result<ProjectPercentiles> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(n)
            WHERE (n:File OR n:Function) AND n.pagerank IS NOT NULL
            WITH collect(toFloat(n.pagerank)) AS prs, collect(toFloat(COALESCE(n.betweenness, 0.0))) AS bws
            WITH prs, bws,
                 apoc.coll.sort(prs) AS sorted_pr,
                 apoc.coll.sort(bws) AS sorted_bw
            WITH sorted_pr, sorted_bw, size(sorted_pr) AS cnt,
                 reduce(s = 0.0, x IN bws | s + x) / size(bws) AS bw_mean
            WITH sorted_pr, sorted_bw, cnt, bw_mean,
                 sorted_pr[toInteger(cnt * 0.5)] AS pr_p50,
                 sorted_pr[toInteger(cnt * 0.8)] AS pr_p80,
                 sorted_pr[toInteger(cnt * 0.95)] AS pr_p95,
                 sorted_bw[toInteger(cnt * 0.5)] AS bw_p50,
                 sorted_bw[toInteger(cnt * 0.8)] AS bw_p80,
                 sorted_bw[toInteger(cnt * 0.95)] AS bw_p95
            WITH *, reduce(s = 0.0, x IN sorted_bw | s + (x - bw_mean) * (x - bw_mean)) / cnt AS bw_var
            RETURN pr_p50, pr_p80, pr_p95, bw_p50, bw_p80, bw_p95, bw_mean, sqrt(bw_var) AS bw_stddev
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            Ok(ProjectPercentiles {
                pagerank_p50: row.get::<f64>("pr_p50").unwrap_or(0.0),
                pagerank_p80: row.get::<f64>("pr_p80").unwrap_or(0.0),
                pagerank_p95: row.get::<f64>("pr_p95").unwrap_or(0.0),
                betweenness_p50: row.get::<f64>("bw_p50").unwrap_or(0.0),
                betweenness_p80: row.get::<f64>("bw_p80").unwrap_or(0.0),
                betweenness_p95: row.get::<f64>("bw_p95").unwrap_or(0.0),
                betweenness_mean: row.get::<f64>("bw_mean").unwrap_or(0.0),
                betweenness_stddev: row.get::<f64>("bw_stddev").unwrap_or(0.0),
            })
        } else {
            // No data — return zeroes
            Ok(ProjectPercentiles {
                pagerank_p50: 0.0,
                pagerank_p80: 0.0,
                pagerank_p95: 0.0,
                betweenness_p50: 0.0,
                betweenness_p80: 0.0,
                betweenness_p95: 0.0,
                betweenness_mean: 0.0,
                betweenness_stddev: 0.0,
            })
        }
    }

    /// Get top N files by betweenness centrality (bridge files).
    pub async fn get_top_bridges_by_betweenness(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<BridgeFile>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.betweenness IS NOT NULL
            RETURN f.path AS path, f.betweenness AS betweenness,
                   f.community_label AS community_label
            ORDER BY f.betweenness DESC
            LIMIT $limit
            "#,
        )
        .param("pid", project_id.to_string())
        .param("limit", limit as i64);

        let rows = self.execute_with_params(q).await?;
        let mut bridges = Vec::new();
        for row in &rows {
            bridges.push(BridgeFile {
                path: row.get::<String>("path").unwrap_or_default(),
                betweenness: row.get::<f64>("betweenness").unwrap_or(0.0),
                community_label: row.get::<String>("community_label").ok(),
            });
        }
        Ok(bridges)
    }

    // ========================================================================
    // Bulk graph extraction (for graph analytics)
    // ========================================================================

    /// Get all IMPORTS edges between files in a project as (source_path, target_path) pairs.
    pub async fn get_project_import_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)-[:IMPORTS]->(f2:File)<-[:CONTAINS]-(p)
            RETURN f1.path AS source, f2.path AS target
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(source), Ok(target)) =
                (row.get::<String>("source"), row.get::<String>("target"))
            {
                edges.push((source, target));
            }
        }

        Ok(edges)
    }

    /// Get all CALLS edges between functions in a project as (caller_id, callee_id) pairs.
    /// Scoped to the same project (no cross-project calls).
    pub async fn get_project_call_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(:File)-[:CONTAINS]->(f1:Function)-[:CALLS]->(f2:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p)
            RETURN f1.id AS source, f2.id AS target
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(source), Ok(target)) =
                (row.get::<String>("source"), row.get::<String>("target"))
            {
                edges.push((source, target));
            }
        }

        Ok(edges)
    }

    /// Batch-update analytics scores on File nodes via UNWIND.
    pub async fn batch_update_file_analytics(
        &self,
        updates: &[crate::graph::models::FileAnalyticsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        // Chunk to avoid overloading Neo4j heap (tx_state = ON_HEAP)
        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), u.path.clone().into());
                    m.insert("pagerank".into(), u.pagerank.into());
                    m.insert("betweenness".into(), u.betweenness.into());
                    m.insert("community_id".into(), (u.community_id as i64).into());
                    m.insert("community_label".into(), u.community_label.clone().into());
                    m.insert(
                        "clustering_coefficient".into(),
                        u.clustering_coefficient.into(),
                    );
                    m.insert("component_id".into(), (u.component_id as i64).into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.pagerank = u.pagerank,
                    f.betweenness = u.betweenness,
                    f.community_id = u.community_id,
                    f.community_label = u.community_label,
                    f.clustering_coefficient = u.clustering_coefficient,
                    f.component_id = u.component_id,
                    f.analytics_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Batch-update analytics scores on Function nodes via parameterized UNWIND.
    ///
    /// Uses chunking (1000 items max per transaction) to avoid heap pressure
    /// and GC death spirals. The query plan is cached by Neo4j since the
    /// Cypher string is constant (parameterized via $items).
    pub async fn batch_update_function_analytics(
        &self,
        updates: &[crate::graph::models::FunctionAnalyticsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("id".into(), u.id.clone().into());
                    m.insert("pagerank".into(), u.pagerank.into());
                    m.insert("betweenness".into(), u.betweenness.into());
                    m.insert("community_id".into(), (u.community_id as i64).into());
                    m.insert(
                        "clustering_coefficient".into(),
                        u.clustering_coefficient.into(),
                    );
                    m.insert("component_id".into(), (u.component_id as i64).into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:Function {id: u.id})
                SET f.pagerank = u.pagerank,
                    f.betweenness = u.betweenness,
                    f.community_id = u.community_id,
                    f.clustering_coefficient = u.clustering_coefficient,
                    f.component_id = u.component_id,
                    f.analytics_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Batch-update **fabric** analytics scores on File nodes.
    ///
    /// Writes to `fabric_*` properties, keeping the existing code-only scores
    /// (`pagerank`, `betweenness`, `community_id`) untouched.
    pub async fn batch_update_fabric_file_analytics(
        &self,
        updates: &[crate::graph::models::FabricFileAnalyticsUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), u.path.clone().into());
                    m.insert("fabric_pagerank".into(), u.fabric_pagerank.into());
                    m.insert("fabric_betweenness".into(), u.fabric_betweenness.into());
                    m.insert(
                        "fabric_community_id".into(),
                        (u.fabric_community_id as i64).into(),
                    );
                    m.insert(
                        "fabric_community_label".into(),
                        u.fabric_community_label.clone().into(),
                    );
                    m.insert(
                        "fabric_clustering_coefficient".into(),
                        u.fabric_clustering_coefficient.into(),
                    );
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.fabric_pagerank = u.fabric_pagerank,
                    f.fabric_betweenness = u.fabric_betweenness,
                    f.fabric_community_id = u.fabric_community_id,
                    f.fabric_community_label = u.fabric_community_label,
                    f.fabric_clustering_coefficient = u.fabric_clustering_coefficient,
                    f.fabric_analytics_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Get SYNAPSE edges bridged from Note-level to File-level for the GDS graph.
    ///
    /// Bridges the neural SYNAPSE connections between Notes to the Files they are
    /// linked to, producing (source_file_path, target_file_path, avg_weight) tuples.
    /// Only includes synapses with weight >= 0.3 (filters weak connections).
    pub async fn get_project_synapse_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, f64)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)
            MATCH (n1:Note)-[:LINKED_TO]->(f1)
            MATCH (n1)-[s:SYNAPSE]->(n2:Note)
            MATCH (n2)-[:LINKED_TO]->(f2:File)
            WHERE f1 <> f2 AND s.weight >= 0.3
            RETURN DISTINCT f1.path AS source, f2.path AS target, avg(s.weight) AS weight
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut edges = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(source), Ok(target), Ok(weight)) = (
                row.get::<String>("source"),
                row.get::<String>("target"),
                row.get::<f64>("weight"),
            ) {
                edges.push((source, target, weight));
            }
        }

        Ok(edges)
    }

    /// Get neural network metrics for a project's SYNAPSE layer.
    ///
    /// Returns aggregate statistics about the note-level neural connections:
    /// active synapse count, average energy, weak synapse ratio, and dead note count.
    pub async fn get_neural_metrics(&self, project_id: Uuid) -> Result<NeuralMetrics> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (n:Note)-[:LINKED_TO]->(f)
            WITH count(DISTINCT n) AS total_notes
            OPTIONAL MATCH (n1:Note)-[s:SYNAPSE]->(n2:Note) WHERE s.weight >= 0.1
            WITH total_notes, count(s) AS active_synapses, avg(s.weight) AS avg_weight,
                 sum(CASE WHEN s.weight < 0.3 THEN 1 ELSE 0 END) AS weak_synapses
            OPTIONAL MATCH (n:Note) WHERE n.energy IS NOT NULL AND n.energy < 0.05
            WITH total_notes, active_synapses, coalesce(avg_weight, 0) AS avg_weight,
                 coalesce(weak_synapses, 0) AS weak_synapses, count(n) AS dead_notes
            RETURN active_synapses, avg_weight AS avg_energy,
                   CASE WHEN active_synapses > 0 THEN toFloat(weak_synapses) / active_synapses ELSE 0.0 END AS weak_synapses_ratio,
                   dead_notes AS dead_notes_count
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            Ok(NeuralMetrics {
                active_synapses: row.get::<i64>("active_synapses").unwrap_or(0),
                avg_energy: row.get::<f64>("avg_energy").unwrap_or(0.0),
                weak_synapses_ratio: row.get::<f64>("weak_synapses_ratio").unwrap_or(0.0),
                dead_notes_count: row.get::<i64>("dead_notes_count").unwrap_or(0),
            })
        } else {
            Ok(NeuralMetrics {
                active_synapses: 0,
                avg_energy: 0.0,
                weak_synapses_ratio: 0.0,
                dead_notes_count: 0,
            })
        }
    }

    // ========================================================================
    // T5.5 — Churn score (commit frequency per file)
    // ========================================================================

    /// Compute churn metrics per file via TOUCHES relations.
    ///
    /// For each file in the project, counts distinct commits touching it,
    /// sums additions+deletions, and counts co-change partners. The churn_score
    /// is normalized 0–1 by dividing by the max commit_count across the project.
    pub async fn compute_churn_scores(&self, project_id: Uuid) -> Result<Vec<FileChurnScore>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (c:Commit)-[t:TOUCHES]->(f)
            WITH f,
                 count(DISTINCT c) AS commit_count,
                 COALESCE(sum(t.additions), 0) + COALESCE(sum(t.deletions), 0) AS total_churn
            OPTIONAL MATCH (f)-[:CO_CHANGED]->()
            WITH f, commit_count, total_churn, count(*) AS co_change_raw
            // co_change_count is 0 when there are no CO_CHANGED rels
            WITH f, commit_count, total_churn,
                 CASE WHEN commit_count = 0 THEN 0 ELSE co_change_raw END AS co_change_count
            ORDER BY commit_count DESC
            RETURN f.path AS path, commit_count, total_churn, co_change_count
            "#,
        )
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Find max commit_count for normalization
        let max_commits = rows
            .iter()
            .filter_map(|r| r.get::<i64>("commit_count").ok())
            .max()
            .unwrap_or(1)
            .max(1);

        let scores: Vec<FileChurnScore> = rows
            .iter()
            .filter_map(|row| {
                let path = row.get::<String>("path").ok()?;
                let commit_count = row.get::<i64>("commit_count").unwrap_or(0);
                let total_churn = row.get::<i64>("total_churn").unwrap_or(0);
                let co_change_count = row.get::<i64>("co_change_count").unwrap_or(0);
                let churn_score = (commit_count as f64 / max_commits as f64).min(1.0);

                Some(FileChurnScore {
                    path,
                    commit_count,
                    total_churn,
                    co_change_count,
                    churn_score,
                })
            })
            .collect();

        Ok(scores)
    }

    /// Batch-write churn scores to File nodes.
    ///
    /// Sets `churn_score` (normalized 0–1), `commit_count`, and `co_change_count`
    /// properties on the matched File nodes.
    pub async fn batch_update_churn_scores(&self, updates: &[FileChurnScore]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), u.path.clone().into());
                    m.insert("churn_score".into(), u.churn_score.into());
                    m.insert("commit_count".into(), u.commit_count.into());
                    m.insert("co_change_count".into(), u.co_change_count.into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.churn_score = u.churn_score,
                    f.commit_count = u.commit_count,
                    f.co_change_count = u.co_change_count,
                    f.churn_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    // ========================================================================
    // T5.6 — Knowledge density score
    // ========================================================================

    /// Compute knowledge density per file based on linked notes and decisions.
    ///
    /// The raw density is `note_count + decision_count * 2`, then normalized
    /// using min-max scaling across the project. Returns 0 if no notes anywhere.
    pub async fn compute_knowledge_density(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<FileKnowledgeDensity>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (n:Note)-[:LINKED_TO]->(f)
            WHERE n.status <> 'obsolete'
            WITH f, count(DISTINCT n) AS note_count,
                 sum(COALESCE(n.energy, 0.5)) AS energy_sum
            OPTIONAL MATCH (d:Decision)-[:AFFECTS]->(f)
            WITH f, note_count, count(DISTINCT d) AS decision_count, energy_sum
            WITH f, note_count, decision_count,
                 toInteger(energy_sum + decision_count * 2) AS raw_density
            RETURN f.path AS path, note_count, decision_count, raw_density
            ORDER BY raw_density DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Min-max normalization of raw_density
        let densities: Vec<i64> = rows
            .iter()
            .filter_map(|r| r.get::<i64>("raw_density").ok())
            .collect();
        let max_density = densities.iter().copied().max().unwrap_or(0);
        let min_density = densities.iter().copied().min().unwrap_or(0);
        let range = (max_density - min_density).max(1); // avoid division by zero

        let scores: Vec<FileKnowledgeDensity> = rows
            .iter()
            .filter_map(|row| {
                let path = row.get::<String>("path").ok()?;
                let note_count = row.get::<i64>("note_count").unwrap_or(0);
                let decision_count = row.get::<i64>("decision_count").unwrap_or(0);
                let raw_density = row.get::<i64>("raw_density").unwrap_or(0);

                // If max_density is 0, no notes anywhere -> all densities are 0
                let knowledge_density = if max_density == 0 {
                    0.0
                } else {
                    ((raw_density - min_density) as f64 / range as f64).min(1.0)
                };

                Some(FileKnowledgeDensity {
                    path,
                    note_count,
                    decision_count,
                    knowledge_density,
                })
            })
            .collect();

        Ok(scores)
    }

    /// Batch-write knowledge density scores to File nodes.
    pub async fn batch_update_knowledge_density(
        &self,
        updates: &[FileKnowledgeDensity],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), u.path.clone().into());
                    m.insert("knowledge_density".into(), u.knowledge_density.into());
                    m.insert("note_count".into(), u.note_count.into());
                    m.insert("decision_count".into(), u.decision_count.into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.knowledge_density = u.knowledge_density,
                    f.note_count = u.note_count,
                    f.decision_count = u.decision_count,
                    f.knowledge_density_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    // ========================================================================
    // T5.7 — Risk score composite
    // ========================================================================

    /// Compute composite risk scores for all files in a project.
    ///
    /// Reads pre-computed properties from File nodes:
    /// - `fabric_pagerank` (structural importance)
    /// - `churn_score` (change frequency)
    /// - `knowledge_density` (documentation coverage)
    /// - `fabric_betweenness` (bridge/bottleneck role)
    ///
    /// Formula: `risk = 0.3 * pagerank_norm + 0.3 * churn + 0.25 * (1 - density) + 0.15 * betweenness_norm`
    pub async fn compute_risk_scores(&self, project_id: Uuid) -> Result<Vec<FileRiskScore>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WITH f,
                 COALESCE(f.fabric_pagerank, f.pagerank, 0.0) AS pr,
                 COALESCE(f.churn_score, 0.0) AS churn,
                 COALESCE(f.knowledge_density, 0.0) AS density,
                 COALESCE(f.fabric_betweenness, f.betweenness, 0.0) AS bw
            WITH collect({
                 path: f.path, pr: pr, churn: churn, density: density, bw: bw
                 }) AS files,
                 max(pr) AS max_pr, max(bw) AS max_bw
            UNWIND files AS file
            WITH file, max_pr, max_bw,
                 CASE WHEN max_pr > 0 THEN file.pr / max_pr ELSE 0.0 END AS pr_norm,
                 CASE WHEN max_bw > 0 THEN file.bw / max_bw ELSE 0.0 END AS bw_norm,
                 file.churn AS churn,
                 file.density AS density
            WITH file.path AS path,
                 pr_norm, churn, density, bw_norm,
                 0.3 * pr_norm + 0.3 * churn + 0.25 * (1.0 - density) + 0.15 * bw_norm AS risk
            RETURN path, risk, pr_norm AS pagerank, churn, 1.0 - density AS knowledge_gap, bw_norm AS betweenness
            ORDER BY risk DESC
            "#,
        )
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        let scores: Vec<FileRiskScore> = rows
            .iter()
            .filter_map(|row| {
                let path = row.get::<String>("path").ok()?;
                let risk_score = row.get::<f64>("risk").unwrap_or(0.0);
                let pagerank = row.get::<f64>("pagerank").unwrap_or(0.0);
                let churn = row.get::<f64>("churn").unwrap_or(0.0);
                let knowledge_gap = row.get::<f64>("knowledge_gap").unwrap_or(0.0);
                let betweenness = row.get::<f64>("betweenness").unwrap_or(0.0);

                let risk_level = if risk_score >= 0.75 {
                    "critical"
                } else if risk_score >= 0.5 {
                    "high"
                } else if risk_score >= 0.25 {
                    "medium"
                } else {
                    "low"
                }
                .to_string();

                Some(FileRiskScore {
                    path,
                    risk_score,
                    risk_level,
                    factors: RiskFactors {
                        pagerank,
                        churn,
                        knowledge_gap,
                        betweenness,
                    },
                })
            })
            .collect();

        Ok(scores)
    }

    /// Batch-write composite risk scores to File nodes.
    pub async fn batch_update_risk_scores(&self, updates: &[FileRiskScore]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in updates.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|u| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), u.path.clone().into());
                    m.insert("risk_score".into(), u.risk_score.into());
                    m.insert("risk_level".into(), u.risk_level.clone().into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.risk_score = u.risk_score,
                    f.risk_level = u.risk_level,
                    f.risk_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    // ========================================================================
    // Helpers for enriched health endpoint (read pre-computed properties)
    // ========================================================================

    /// Get top N files by churn_score (pre-computed on File nodes).
    /// Returns empty vec if churn_score has not been computed yet.
    pub async fn get_top_hotspots(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<FileChurnScore>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.churn_score IS NOT NULL
            RETURN f.path AS path,
                   COALESCE(f.commit_count, 0) AS commit_count,
                   0 AS total_churn,
                   COALESCE(f.co_change_count, 0) AS co_change_count,
                   f.churn_score AS churn_score
            ORDER BY f.churn_score DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let rows = self.execute_with_params(q).await?;
        let mut results = Vec::new();
        for row in &rows {
            if let Ok(path) = row.get::<String>("path") {
                results.push(FileChurnScore {
                    path,
                    commit_count: row.get::<i64>("commit_count").unwrap_or(0),
                    total_churn: row.get::<i64>("total_churn").unwrap_or(0),
                    co_change_count: row.get::<i64>("co_change_count").unwrap_or(0),
                    churn_score: row.get::<f64>("churn_score").unwrap_or(0.0),
                });
            }
        }
        Ok(results)
    }

    /// Get top N files with lowest knowledge_density (knowledge gaps).
    /// Returns empty vec if knowledge_density has not been computed yet.
    pub async fn get_top_knowledge_gaps(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<FileKnowledgeDensity>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.knowledge_density IS NOT NULL
            RETURN f.path AS path,
                   COALESCE(f.note_count, 0) AS note_count,
                   COALESCE(f.decision_count, 0) AS decision_count,
                   f.knowledge_density AS knowledge_density
            ORDER BY f.knowledge_density ASC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let rows = self.execute_with_params(q).await?;
        let mut results = Vec::new();
        for row in &rows {
            if let Ok(path) = row.get::<String>("path") {
                results.push(FileKnowledgeDensity {
                    path,
                    note_count: row.get::<i64>("note_count").unwrap_or(0),
                    decision_count: row.get::<i64>("decision_count").unwrap_or(0),
                    knowledge_density: row.get::<f64>("knowledge_density").unwrap_or(0.0),
                });
            }
        }
        Ok(results)
    }

    /// Get risk assessment summary stats for a project (pre-computed on File nodes).
    pub async fn get_risk_summary(&self, project_id: Uuid) -> Result<serde_json::Value> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.risk_score IS NOT NULL
            WITH count(f) AS total,
                 avg(f.risk_score) AS avg_risk,
                 max(f.risk_score) AS max_risk,
                 sum(CASE WHEN f.risk_level = 'critical' THEN 1 ELSE 0 END) AS critical_count,
                 sum(CASE WHEN f.risk_level = 'high' THEN 1 ELSE 0 END) AS high_count,
                 sum(CASE WHEN f.risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_count,
                 sum(CASE WHEN f.risk_level = 'low' THEN 1 ELSE 0 END) AS low_count
            RETURN total, avg_risk, max_risk, critical_count, high_count, medium_count, low_count
            "#,
        )
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let total = row.get::<i64>("total").unwrap_or(0);
            if total == 0 {
                return Ok(serde_json::json!(null));
            }
            Ok(serde_json::json!({
                "files_assessed": total,
                "avg_risk_score": row.get::<f64>("avg_risk").unwrap_or(0.0),
                "max_risk_score": row.get::<f64>("max_risk").unwrap_or(0.0),
                "critical_count": row.get::<i64>("critical_count").unwrap_or(0),
                "high_count": row.get::<i64>("high_count").unwrap_or(0),
                "medium_count": row.get::<i64>("medium_count").unwrap_or(0),
                "low_count": row.get::<i64>("low_count").unwrap_or(0),
            }))
        } else {
            Ok(serde_json::json!(null))
        }
    }
}
