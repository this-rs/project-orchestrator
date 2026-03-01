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
                 collect(f.path) AS all_paths,
                 size([x IN collect(DISTINCT f.cc_wl_hash) WHERE x IS NOT NULL AND x <> 0]) AS unique_fps
            ORDER BY file_count DESC
            RETURN cid, label, file_count,
                   [p IN all_paths | p][..3] AS key_files,
                   unique_fps
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

            let unique_fingerprints = row.get::<i64>("unique_fps").unwrap_or(0) as usize;

            communities.push(CommunityRow {
                community_id,
                community_label,
                file_count,
                key_files,
                unique_fingerprints,
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

    /// Get all EXTENDS edges between structs/classes in a project as (child_file, parent_file) pairs.
    /// Returns file-level edges so the graph analytics engine can weight inter-file coupling.
    pub async fn get_project_extends_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s1:Struct)-[:EXTENDS]->(s2:Struct)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p)
            WHERE f1 <> f2
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

    /// Get all IMPLEMENTS edges between structs and traits in a project as (struct_file, trait_file) pairs.
    /// Returns file-level edges so the graph analytics engine can weight inter-file coupling.
    pub async fn get_project_implements_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s:Struct)-[:IMPLEMENTS]->(t:Trait)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p)
            WHERE f1 <> f2
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

    /// Batch-update structural DNA vectors on File nodes.
    ///
    /// Uses UNWIND in chunks of 1000 to write `structural_dna` as a list of floats.
    /// DNA = K-dimensional distance vector to PageRank anchor nodes, normalized [0,1].
    pub async fn batch_update_structural_dna(
        &self,
        updates: &[crate::graph::models::StructuralDnaUpdate],
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
                    // Convert Vec<f64> to BoltType list
                    let dna_list: Vec<neo4rs::BoltType> = u.dna.iter().map(|&v| v.into()).collect();
                    m.insert(
                        "dna".into(),
                        neo4rs::BoltType::List(neo4rs::BoltList::from(dna_list)),
                    );
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.structural_dna = u.dna,
                    f.structural_dna_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Batch-update structural fingerprint vectors on File nodes.
    ///
    /// Uses UNWIND in chunks of 1000 to write `structural_fingerprint` as a list of floats.
    /// Fingerprint = 17-dimensional universal feature vector, project-independent.
    pub async fn batch_update_structural_fingerprints(
        &self,
        updates: &[crate::graph::models::StructuralFingerprintUpdate],
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
                    let fp_list: Vec<neo4rs::BoltType> =
                        u.fingerprint.iter().map(|&v| v.into()).collect();
                    m.insert(
                        "fingerprint".into(),
                        neo4rs::BoltType::List(neo4rs::BoltList::from(fp_list)),
                    );
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS u
                MATCH (f:File {path: u.path})
                SET f.structural_fingerprint = u.fingerprint,
                    f.structural_fingerprint_updated_at = datetime()
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Read structural fingerprint vectors for all File nodes in a project.
    ///
    /// Returns (file_path, fingerprint_vector) pairs.
    pub async fn get_project_structural_fingerprints(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.structural_fingerprint IS NOT NULL
            RETURN f.path AS path, f.structural_fingerprint AS fingerprint
            "#,
        )
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut fp_list = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(path), Ok(fp)) = (
                row.get::<String>("path"),
                row.get::<Vec<f64>>("fingerprint"),
            ) {
                fp_list.push((path, fp));
            }
        }

        Ok(fp_list)
    }

    /// Read all file signals needed for multi-signal structural similarity.
    ///
    /// Single query returns fingerprint, WL hash, and function count for each file.
    /// Scoped by project_id. Only returns files that have a computed fingerprint.
    pub async fn get_project_file_signals(
        &self,
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::FileSignalRecord>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.structural_fingerprint IS NOT NULL
            OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
            WITH f, count(fn) AS func_count
            RETURN f.path AS path,
                   f.structural_fingerprint AS fingerprint,
                   COALESCE(f.cc_wl_hash, 0) AS wl_hash,
                   func_count AS function_count
            "#,
        )
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut records = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(path) = row.get::<String>("path") {
                let fingerprint: Vec<f64> = row.get("fingerprint").unwrap_or_default();
                let wl_hash = row.get::<i64>("wl_hash").unwrap_or(0) as u64;
                let function_count = row.get::<i64>("function_count").unwrap_or(0) as usize;
                records.push(crate::graph::models::FileSignalRecord {
                    path,
                    fingerprint,
                    wl_hash,
                    function_count,
                });
            }
        }

        Ok(records)
    }

    /// Write predicted missing links for a project.
    ///
    /// First removes old PREDICTED_LINK edges for this project, then creates new ones.
    /// Each prediction becomes a relationship: (source)-[:PREDICTED_LINK]->(target)
    /// with properties: plausibility, suggested_relation, signals (as JSON string), computed_at.
    pub async fn write_predicted_links(
        &self,
        project_id: &str,
        links: &[crate::graph::models::LinkPrediction],
    ) -> Result<()> {
        // 1. Remove old predicted links for this project
        let delete_q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)-[r:PREDICTED_LINK]->()
            DELETE r
            "#,
        )
        .param("project_id", project_id.to_string());
        self.graph.run(delete_q).await?;

        if links.is_empty() {
            return Ok(());
        }

        // 2. Create new predicted links in batch
        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = links
            .iter()
            .map(|l| {
                let mut m = std::collections::HashMap::new();
                m.insert("source".into(), l.source.clone().into());
                m.insert("target".into(), l.target.clone().into());
                m.insert("plausibility".into(), l.plausibility.into());
                m.insert(
                    "suggested_relation".into(),
                    l.suggested_relation.clone().into(),
                );
                // Store signals as a JSON string for simplicity
                let signals_str: String = l
                    .signals
                    .iter()
                    .map(|(name, val)| format!("{}={:.4}", name, val))
                    .collect::<Vec<_>>()
                    .join(",");
                m.insert("signals".into(), signals_str.into());
                m
            })
            .collect();

        let create_q = query(
            r#"
            UNWIND $items AS item
            MATCH (s {path: item.source}), (t {path: item.target})
            MERGE (s)-[r:PREDICTED_LINK]->(t)
            SET r.plausibility = item.plausibility,
                r.suggested_relation = item.suggested_relation,
                r.signals = item.signals,
                r.computed_at = datetime()
            "#,
        )
        .param("items", items);

        self.graph.run(create_q).await?;

        Ok(())
    }

    /// Read structural DNA vectors for all File nodes in a project.
    ///
    /// Returns (file_path, dna_vector) pairs for files that have DNA computed.
    pub async fn get_project_structural_dna(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.structural_dna IS NOT NULL
            RETURN f.path AS path, f.structural_dna AS dna
            "#,
        )
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut dna_list = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(path), Ok(dna)) = (row.get::<String>("path"), row.get::<Vec<f64>>("dna")) {
                dna_list.push((path, dna));
            }
        }

        Ok(dna_list)
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

    /// Batch-write context cards as cc_* properties on File nodes.
    ///
    /// Uses UNWIND with chunks of 1000 for efficient batch writes.
    /// DNA vectors and co_changers are stored as serialized JSON strings
    /// since Neo4j doesn't support nested arrays in UNWIND.
    pub async fn batch_save_context_cards(
        &self,
        cards: &[crate::graph::models::ContextCard],
    ) -> Result<()> {
        if cards.is_empty() {
            return Ok(());
        }

        const CHUNK_SIZE: usize = 1000;

        for chunk in cards.chunks(CHUNK_SIZE) {
            let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = chunk
                .iter()
                .map(|c| {
                    let mut m = std::collections::HashMap::new();
                    m.insert("path".into(), c.path.clone().into());
                    m.insert("cc_pagerank".into(), c.cc_pagerank.into());
                    m.insert("cc_betweenness".into(), c.cc_betweenness.into());
                    m.insert("cc_clustering".into(), c.cc_clustering.into());
                    m.insert("cc_community_id".into(), (c.cc_community_id as i64).into());
                    m.insert(
                        "cc_community_label".into(),
                        c.cc_community_label.clone().into(),
                    );
                    m.insert("cc_imports_out".into(), (c.cc_imports_out as i64).into());
                    m.insert("cc_imports_in".into(), (c.cc_imports_in as i64).into());
                    m.insert("cc_calls_out".into(), (c.cc_calls_out as i64).into());
                    m.insert("cc_calls_in".into(), (c.cc_calls_in as i64).into());
                    // DNA vector stored as JSON string (nested arrays not supported in UNWIND)
                    let dna_json = serde_json::to_string(&c.cc_structural_dna).unwrap_or_default();
                    m.insert("cc_structural_dna".into(), dna_json.into());
                    m.insert("cc_wl_hash".into(), (c.cc_wl_hash as i64).into());
                    // co_changers stored as JSON string
                    let co_changers_json =
                        serde_json::to_string(&c.cc_co_changers_top5).unwrap_or_default();
                    m.insert("cc_co_changers_top5".into(), co_changers_json.into());
                    // Fingerprint vector stored as JSON string (nested arrays not supported in UNWIND)
                    let fp_json = serde_json::to_string(&c.cc_fingerprint).unwrap_or_default();
                    m.insert("cc_fingerprint".into(), fp_json.into());
                    m.insert("cc_version".into(), (c.cc_version as i64).into());
                    m.insert("cc_computed_at".into(), c.cc_computed_at.clone().into());
                    m
                })
                .collect();

            let q = query(
                r#"
                UNWIND $items AS row
                MATCH (f:File {path: row.path})
                SET f.cc_pagerank = row.cc_pagerank,
                    f.cc_betweenness = row.cc_betweenness,
                    f.cc_clustering = row.cc_clustering,
                    f.cc_community_id = row.cc_community_id,
                    f.cc_community_label = row.cc_community_label,
                    f.cc_imports_out = row.cc_imports_out,
                    f.cc_imports_in = row.cc_imports_in,
                    f.cc_calls_out = row.cc_calls_out,
                    f.cc_calls_in = row.cc_calls_in,
                    f.cc_structural_dna = row.cc_structural_dna,
                    f.cc_wl_hash = row.cc_wl_hash,
                    f.cc_co_changers_top5 = row.cc_co_changers_top5,
                    f.cc_fingerprint = row.cc_fingerprint,
                    f.cc_version = row.cc_version,
                    f.cc_computed_at = row.cc_computed_at
                "#,
            )
            .param("items", items);

            self.graph.run(q).await?;
        }

        Ok(())
    }

    /// Invalidate context cards for given file paths and their 1-hop neighbors.
    ///
    /// Sets `cc_version = -1` on the target files and any direct neighbor
    /// connected via IMPORTS or CALLS relationships. This ensures stale cards
    /// are recomputed on next analytics run.
    pub async fn invalidate_context_cards(&self, paths: &[String], project_id: &str) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let path_list: Vec<neo4rs::BoltType> = paths.iter().map(|p| p.clone().into()).collect();

        let q = query(
            r#"
            UNWIND $paths AS path
            MATCH (f:File {path: path, project_id: $project_id})
            SET f.cc_version = -1
            WITH f
            OPTIONAL MATCH (f)-[:IMPORTS|CALLS]-(neighbor:File)
            SET neighbor.cc_version = -1
            "#,
        )
        .param("paths", path_list)
        .param("project_id", project_id);

        self.graph.run(q).await?;

        Ok(())
    }

    /// Read a context card from Neo4j cc_* properties for a single file.
    ///
    /// Returns `None` if the file doesn't exist or has no cc_* properties.
    /// The caller should check `cc_version`: if -1, the card is stale.
    pub async fn get_context_card(
        &self,
        path: &str,
        project_id: &str,
    ) -> Result<Option<crate::graph::models::ContextCard>> {
        let q = query(
            r#"
            MATCH (f:File {path: $path, project_id: $project_id})
            WHERE f.cc_version IS NOT NULL
            RETURN f.path AS path,
                   COALESCE(f.cc_pagerank, 0.0) AS cc_pagerank,
                   COALESCE(f.cc_betweenness, 0.0) AS cc_betweenness,
                   COALESCE(f.cc_clustering, 0.0) AS cc_clustering,
                   COALESCE(f.cc_community_id, 0) AS cc_community_id,
                   COALESCE(f.cc_community_label, '') AS cc_community_label,
                   COALESCE(f.cc_imports_out, 0) AS cc_imports_out,
                   COALESCE(f.cc_imports_in, 0) AS cc_imports_in,
                   COALESCE(f.cc_calls_out, 0) AS cc_calls_out,
                   COALESCE(f.cc_calls_in, 0) AS cc_calls_in,
                   COALESCE(f.cc_structural_dna, '[]') AS cc_structural_dna,
                   COALESCE(f.cc_wl_hash, 0) AS cc_wl_hash,
                   COALESCE(f.cc_co_changers_top5, '[]') AS cc_co_changers_top5,
                   COALESCE(f.cc_fingerprint, '[]') AS cc_fingerprint,
                   COALESCE(f.cc_version, 0) AS cc_version,
                   COALESCE(f.cc_computed_at, '') AS cc_computed_at
            "#,
        )
        .param("path", path)
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let dna_json: String = row.get("cc_structural_dna").unwrap_or_default();
            let co_changers_json: String = row.get("cc_co_changers_top5").unwrap_or_default();

            Ok(Some(crate::graph::models::ContextCard {
                path: row.get("path").unwrap_or_default(),
                cc_pagerank: row.get("cc_pagerank").unwrap_or(0.0),
                cc_betweenness: row.get("cc_betweenness").unwrap_or(0.0),
                cc_clustering: row.get("cc_clustering").unwrap_or(0.0),
                cc_community_id: row.get::<i64>("cc_community_id").unwrap_or(0) as u32,
                cc_community_label: row.get("cc_community_label").unwrap_or_default(),
                cc_imports_out: row.get::<i64>("cc_imports_out").unwrap_or(0) as usize,
                cc_imports_in: row.get::<i64>("cc_imports_in").unwrap_or(0) as usize,
                cc_calls_out: row.get::<i64>("cc_calls_out").unwrap_or(0) as usize,
                cc_calls_in: row.get::<i64>("cc_calls_in").unwrap_or(0) as usize,
                cc_structural_dna: serde_json::from_str(&dna_json).unwrap_or_default(),
                cc_wl_hash: row.get::<i64>("cc_wl_hash").unwrap_or(0) as u64,
                cc_fingerprint: {
                    let fp_json: String = row.get("cc_fingerprint").unwrap_or_default();
                    serde_json::from_str(&fp_json).unwrap_or_default()
                },
                cc_co_changers_top5: serde_json::from_str(&co_changers_json).unwrap_or_default(),
                cc_version: row.get::<i64>("cc_version").unwrap_or(0) as i32,
                cc_computed_at: row.get("cc_computed_at").unwrap_or_default(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Batch-read context cards for multiple files in one query.
    pub async fn get_context_cards_batch(
        &self,
        paths: &[String],
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::ContextCard>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let path_list: Vec<neo4rs::BoltType> = paths.iter().map(|p| p.clone().into()).collect();

        let q = query(
            r#"
            UNWIND $paths AS path
            MATCH (f:File {path: path, project_id: $project_id})
            WHERE f.cc_version IS NOT NULL
            RETURN f.path AS path,
                   COALESCE(f.cc_pagerank, 0.0) AS cc_pagerank,
                   COALESCE(f.cc_betweenness, 0.0) AS cc_betweenness,
                   COALESCE(f.cc_clustering, 0.0) AS cc_clustering,
                   COALESCE(f.cc_community_id, 0) AS cc_community_id,
                   COALESCE(f.cc_community_label, '') AS cc_community_label,
                   COALESCE(f.cc_imports_out, 0) AS cc_imports_out,
                   COALESCE(f.cc_imports_in, 0) AS cc_imports_in,
                   COALESCE(f.cc_calls_out, 0) AS cc_calls_out,
                   COALESCE(f.cc_calls_in, 0) AS cc_calls_in,
                   COALESCE(f.cc_structural_dna, '[]') AS cc_structural_dna,
                   COALESCE(f.cc_wl_hash, 0) AS cc_wl_hash,
                   COALESCE(f.cc_co_changers_top5, '[]') AS cc_co_changers_top5,
                   COALESCE(f.cc_fingerprint, '[]') AS cc_fingerprint,
                   COALESCE(f.cc_version, 0) AS cc_version,
                   COALESCE(f.cc_computed_at, '') AS cc_computed_at
            "#,
        )
        .param("paths", path_list)
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        let mut cards = Vec::new();

        while let Some(row) = result.next().await? {
            let dna_json: String = row.get("cc_structural_dna").unwrap_or_default();
            let co_changers_json: String = row.get("cc_co_changers_top5").unwrap_or_default();

            cards.push(crate::graph::models::ContextCard {
                path: row.get("path").unwrap_or_default(),
                cc_pagerank: row.get("cc_pagerank").unwrap_or(0.0),
                cc_betweenness: row.get("cc_betweenness").unwrap_or(0.0),
                cc_clustering: row.get("cc_clustering").unwrap_or(0.0),
                cc_community_id: row.get::<i64>("cc_community_id").unwrap_or(0) as u32,
                cc_community_label: row.get("cc_community_label").unwrap_or_default(),
                cc_imports_out: row.get::<i64>("cc_imports_out").unwrap_or(0) as usize,
                cc_imports_in: row.get::<i64>("cc_imports_in").unwrap_or(0) as usize,
                cc_calls_out: row.get::<i64>("cc_calls_out").unwrap_or(0) as usize,
                cc_calls_in: row.get::<i64>("cc_calls_in").unwrap_or(0) as usize,
                cc_structural_dna: serde_json::from_str(&dna_json).unwrap_or_default(),
                cc_wl_hash: row.get::<i64>("cc_wl_hash").unwrap_or(0) as u64,
                cc_fingerprint: {
                    let fp_json: String = row.get("cc_fingerprint").unwrap_or_default();
                    serde_json::from_str(&fp_json).unwrap_or_default()
                },
                cc_co_changers_top5: serde_json::from_str(&co_changers_json).unwrap_or_default(),
                cc_version: row.get::<i64>("cc_version").unwrap_or(0) as i32,
                cc_computed_at: row.get("cc_computed_at").unwrap_or_default(),
            });
        }

        Ok(cards)
    }

    /// Find groups of files with identical WL hash (isomorphic neighborhoods).
    /// Groups cc_wl_hash values, filtering to groups with at least `min_group_size` members.
    pub async fn find_isomorphic_groups(
        &self,
        project_id: &str,
        min_group_size: usize,
    ) -> Result<Vec<crate::graph::models::IsomorphicGroup>> {
        let q = query(
            r#"
            MATCH (f:File {project_id: $project_id})
            WHERE f.cc_wl_hash IS NOT NULL AND f.cc_wl_hash <> 0
            WITH f.cc_wl_hash AS wl_hash, COLLECT(f.path) AS members
            WHERE SIZE(members) >= $min_size
            RETURN wl_hash, members
            ORDER BY SIZE(members) DESC
            "#,
        )
        .param("project_id", project_id)
        .param("min_size", min_group_size as i64);

        let mut result = self.graph.execute(q).await?;
        let mut groups = Vec::new();

        while let Some(row) = result.next().await? {
            let wl_hash = row.get::<i64>("wl_hash").unwrap_or(0) as u64;
            let members: Vec<String> = row.get::<Vec<String>>("members").unwrap_or_default();
            let size = members.len();
            groups.push(crate::graph::models::IsomorphicGroup {
                wl_hash,
                members,
                size,
            });
        }

        Ok(groups)
    }

    /// Check if any file in the project has GraIL analytics computed (cc_version).
    /// Returns `true` if at least one file has context cards, `false` otherwise.
    /// Used by staleness check to detect projects that were synced before GraIL
    /// was deployed and need their first analytics computation.
    pub async fn has_context_cards(&self, project_id: &str) -> Result<bool> {
        let q = query(
            r#"
            MATCH (f:File {project_id: $project_id})
            WHERE f.cc_version IS NOT NULL AND f.cc_version > 0
            RETURN count(f) > 0 AS has_cards
            "#,
        )
        .param("project_id", project_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<bool>("has_cards").unwrap_or(false))
        } else {
            Ok(false)
        }
    }
}
