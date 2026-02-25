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
}
