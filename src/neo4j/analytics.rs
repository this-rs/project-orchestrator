//! Neo4j Analytics operations (communities, health, GDS metrics)

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

/// Map a scaffolding level (0-4) to its label and recommended steps description.
pub fn level_info(level: u8) -> (u8, String, String) {
    match level {
        0 => (
            0,
            "L0 — Reflexe".to_string(),
            "5+ steps détaillés avec snippets".to_string(),
        ),
        1 => (
            1,
            "L1 — Associatif".to_string(),
            "3-4 steps guidés".to_string(),
        ),
        2 => (
            2,
            "L2 — Contextuel".to_string(),
            "3-4 steps standard".to_string(),
        ),
        3 => (
            3,
            "L3 — Stratégique".to_string(),
            "2 steps, autonomie élevée".to_string(),
        ),
        _ => (
            4,
            "L4 — Méta-cognitif".to_string(),
            "1 step abstrait, libre décomposition".to_string(),
        ),
    }
}

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

        // WorldModel prediction accuracy (biomimicry T7)
        let prediction_accuracy = self.compute_prediction_accuracy(project_id, 10).await.ok();

        Ok(CodeHealthReport {
            god_functions,
            orphan_files,
            coupling_metrics,
            prediction_accuracy,
        })
    }

    /// WorldModel prediction accuracy (biomimicry T7):
    /// For each of the last N sessions, compute what files the agent discussed,
    /// then check if those files were predictable from the CO_CHANGED graph
    /// of files discussed in the *previous* session.
    pub async fn compute_prediction_accuracy(
        &self,
        project_id: Uuid,
        max_sessions: i64,
    ) -> Result<crate::neo4j::models::PredictionAccuracy> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_CHAT_SESSION]->(s:ChatSession)
            WITH s ORDER BY s.created_at DESC LIMIT $max_sessions
            WITH collect(s) AS sessions
            UNWIND range(0, size(sessions) - 2) AS i
            WITH sessions[i] AS current_session, sessions[i+1] AS prev_session
            MATCH (prev_session)-[:DISCUSSED]->(prev_file:File)
            WITH current_session, collect(DISTINCT prev_file.path) AS prev_files, prev_session
            OPTIONAL MATCH (pf:File)-[:CO_CHANGED]->(predicted:File)
            WHERE pf.path IN prev_files AND predicted.path <> pf.path
            WITH current_session, collect(DISTINCT predicted.path) AS predicted_paths
            MATCH (current_session)-[:DISCUSSED]->(actual_file:File)
            WITH current_session, predicted_paths, collect(DISTINCT actual_file.path) AS actual_paths
            WITH current_session,
                 size(actual_paths) AS total_accessed,
                 size([f IN actual_paths WHERE f IN predicted_paths]) AS hits
            RETURN sum(hits) AS total_hits,
                   sum(total_accessed) AS total_accessed,
                   count(current_session) AS sessions_analyzed
            "#,
        )
        .param("pid", project_id.to_string())
        .param("max_sessions", max_sessions);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let hits = row.get::<i64>("total_hits").unwrap_or(0);
            let total = row.get::<i64>("total_accessed").unwrap_or(0);
            let sessions = row.get::<i64>("sessions_analyzed").unwrap_or(0);
            let accuracy = if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            };
            Ok(crate::neo4j::models::PredictionAccuracy {
                hits,
                total,
                accuracy,
                sessions_analyzed: sessions,
            })
        } else {
            Ok(crate::neo4j::models::PredictionAccuracy {
                hits: 0,
                total: 0,
                accuracy: 0.0,
                sessions_analyzed: 0,
            })
        }
    }

    /// Capture a lightweight maintenance snapshot (biomimicry T11).
    /// 5 queries: health_score proxy, active_synapses, mean_energy, skill_count, note_count.
    pub async fn compute_maintenance_snapshot(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::MaintenanceSnapshot> {
        use chrono::Utc;

        // Q1: health_score proxy — count god functions + orphan files (lower = healthier)
        let q1 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)-[:DEFINES]->(fn:Function)
            WHERE fn.line_count > 100
            WITH count(fn) AS god_fns
            OPTIONAL MATCH (p2:Project {id: $pid})-[:CONTAINS]->(orphan:File)
            WHERE NOT EXISTS { (orphan)-[:IMPORTS]->() }
              AND NOT EXISTS { ()-[:IMPORTS]->(orphan) }
            WITH god_fns, count(orphan) AS orphans
            RETURN god_fns, orphans,
                   CASE WHEN god_fns + orphans = 0 THEN 1.0
                        ELSE 1.0 / (1.0 + god_fns * 0.1 + orphans * 0.05)
                   END AS health_score
            "#,
        )
        .param("pid", project_id.to_string());

        // Q2: active synapses count
        let q2 = query(
            r#"
            MATCH (p:Project {id: $pid})
            OPTIONAL MATCH (p)-[:HAS_NOTE]->(n1:Note)-[s:SYNAPSE]->(n2:Note)
            WHERE s.strength > 0
            RETURN count(s) AS active_synapses
            "#,
        )
        .param("pid", project_id.to_string());

        // Q3: mean energy across notes
        let q3 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_NOTE]->(n:Note)
            WHERE n.status = 'active'
            RETURN avg(coalesce(n.energy, 0.5)) AS mean_energy
            "#,
        )
        .param("pid", project_id.to_string());

        // Q4: active skill count
        let q4 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_SKILL]->(s:Skill)
            WHERE s.status IN ['active', 'emerging']
            RETURN count(s) AS skill_count
            "#,
        )
        .param("pid", project_id.to_string());

        // Q5: active note count
        let q5 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_NOTE]->(n:Note)
            WHERE n.status = 'active'
            RETURN count(n) AS note_count
            "#,
        )
        .param("pid", project_id.to_string());

        // Execute all 5 in parallel
        let (r1, r2, r3, r4, r5) = tokio::join!(
            self.execute_with_params(q1),
            self.execute_with_params(q2),
            self.execute_with_params(q3),
            self.execute_with_params(q4),
            self.execute_with_params(q5),
        );

        let health_score = r1
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<f64>("health_score").ok())
            .unwrap_or(0.5);

        let active_synapses = r2
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<i64>("active_synapses").ok())
            .unwrap_or(0);

        let mean_energy = r3
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<f64>("mean_energy").ok())
            .unwrap_or(0.5);

        let skill_count = r4
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<i64>("skill_count").ok())
            .unwrap_or(0);

        let note_count = r5
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<i64>("note_count").ok())
            .unwrap_or(0);

        Ok(crate::neo4j::models::MaintenanceSnapshot {
            health_score,
            active_synapses,
            mean_energy,
            skill_count,
            note_count,
            captured_at: Utc::now().to_rfc3339(),
        })
    }

    /// Compute the scaffolding level for adaptive task complexity (biomimicry T8).
    /// Combines 4 signals: task_success_rate, avg_frustration, scar_density, homeostasis_pain.
    /// Returns a ScaffoldingLevel (L0-L4) with competence score and metrics.
    pub async fn compute_scaffolding_level(
        &self,
        project_id: Uuid,
        scaffolding_override: Option<u8>,
    ) -> Result<crate::neo4j::models::ScaffoldingLevel> {
        // Q1: task success rate + avg frustration (last 20 tasks)
        let q1 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_PLAN]->(plan)-[:HAS_TASK]->(t:Task)
            WHERE t.status IN ['completed', 'failed']
            WITH t ORDER BY t.updated_at DESC LIMIT 20
            WITH count(t) AS total,
                 sum(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS completed,
                 avg(coalesce(t.frustration_score, 0.0)) AS avg_frust
            RETURN total, completed,
                   CASE WHEN total > 0 THEN toFloat(completed) / total ELSE 1.0 END AS success_rate,
                   avg_frust
            "#,
        )
        .param("pid", project_id.to_string());

        // Q2: scar density (avg scar_intensity across project notes)
        let q2 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_NOTE]->(n:Note)
            WHERE n.status = 'active'
            RETURN avg(coalesce(n.scar_intensity, 0.0)) AS scar_density
            "#,
        )
        .param("pid", project_id.to_string());

        // Execute in parallel
        let (r1, r2) = tokio::join!(self.execute_with_params(q1), self.execute_with_params(q2),);

        let (tasks_analyzed, task_success_rate, avg_frustration) = r1
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .map(|r| {
                let total = r.get::<i64>("total").unwrap_or(0);
                let rate = r.get::<f64>("success_rate").unwrap_or(1.0);
                let frust = r.get::<f64>("avg_frust").unwrap_or(0.0);
                (total, rate, frust)
            })
            .unwrap_or((0, 1.0, 0.0));

        let scar_density = r2
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get::<f64>("scar_density").ok())
            .unwrap_or(0.0);

        // Q3: homeostasis pain (reuse compute_homeostasis)
        let homeostasis_pain = self
            .compute_homeostasis(project_id, None)
            .await
            .map(|h| h.pain_score)
            .unwrap_or(0.0);

        // Composite competence score (0.0 = struggling, 1.0 = expert)
        // Weights: success_rate dominates, frustration/scars/pain are penalties
        let competence_score = (task_success_rate * 0.5
            + (1.0 - avg_frustration) * 0.2
            + (1.0 - scar_density) * 0.15
            + (1.0 - homeostasis_pain) * 0.15)
            .clamp(0.0, 1.0);

        // Map competence to level (L0-L4)
        let (level, label, recommended_steps) = if let Some(ovr) = scaffolding_override {
            let ovr = ovr.min(4);
            level_info(ovr)
        } else {
            let auto_level = if competence_score >= 0.9 {
                4
            } else if competence_score >= 0.75 {
                3
            } else if competence_score >= 0.5 {
                2
            } else if competence_score >= 0.3 {
                1
            } else {
                0
            };
            level_info(auto_level)
        };

        Ok(crate::neo4j::models::ScaffoldingLevel {
            level,
            label,
            recommended_steps,
            task_success_rate,
            avg_frustration,
            scar_density,
            homeostasis_pain,
            competence_score,
            is_overridden: scaffolding_override.is_some(),
            tasks_analyzed,
        })
    }

    /// Detect global stagnation across a project (biomimicry T12).
    /// Checks 4 signals: tasks completed in 48h, avg frustration, energy trend, commits in 48h.
    /// If ≥3 signals triggered → stagnation detected.
    pub async fn detect_global_stagnation(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::StagnationReport> {
        // Q1: tasks completed in last 48h
        let q1 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_PLAN]->(plan)-[:HAS_TASK]->(t:Task)
            WHERE t.status = 'completed'
              AND t.completed_at IS NOT NULL
              AND datetime(t.completed_at) > datetime() - duration('PT48H')
            RETURN count(t) AS cnt
            "#,
        )
        .param("pid", project_id.to_string());

        // Q2: avg frustration on in-progress tasks
        let q2 = query(
            r#"
            MATCH (p:Project {id: $pid})-[:HAS_PLAN]->(plan)-[:HAS_TASK]->(t:Task)
            WHERE t.status = 'in_progress' AND t.frustration IS NOT NULL
            RETURN avg(t.frustration) AS avg_f, count(t) AS cnt
            "#,
        )
        .param("pid", project_id.to_string());

        // Q3: mean note energy (current snapshot)
        let q3 = query(
            r#"
            MATCH (n:Note {project_id: $pid})
            WHERE n.status = 'active' AND n.energy IS NOT NULL
            RETURN avg(n.energy) AS avg_energy, count(n) AS cnt
            "#,
        )
        .param("pid", project_id.to_string());

        // Q4: commits (TOUCHES) in last 48h
        let q4 = query(
            r#"
            MATCH (c:Commit)-[:TOUCHES]->(f:File)
            WHERE EXISTS { MATCH (p:Project {id: $pid})-[:CONTAINS]->(f) }
              AND c.created_at IS NOT NULL
              AND datetime(c.created_at) > datetime() - duration('PT48H')
            RETURN count(DISTINCT c) AS cnt
            "#,
        )
        .param("pid", project_id.to_string());

        let (r1, r2, r3, r4) = tokio::join!(
            self.execute_with_params(q1),
            self.execute_with_params(q2),
            self.execute_with_params(q3),
            self.execute_with_params(q4),
        );

        let tasks_completed_48h = r1
            .unwrap_or_default()
            .first()
            .and_then(|r| r.get::<i64>("cnt").ok())
            .unwrap_or(0);

        let avg_frustration = r2
            .unwrap_or_default()
            .first()
            .and_then(|r| r.get::<f64>("avg_f").ok())
            .unwrap_or(0.0);

        let mean_energy = r3
            .unwrap_or_default()
            .first()
            .and_then(|r| r.get::<f64>("avg_energy").ok())
            .unwrap_or(1.0);

        let commits_48h = r4
            .unwrap_or_default()
            .first()
            .and_then(|r| r.get::<i64>("cnt").ok())
            .unwrap_or(0);

        // Energy trend: below 0.4 = declining (proxy without historical snapshots)
        let energy_trend = mean_energy - 0.5; // negative = below midpoint

        // Count triggered signals
        let mut signals: u8 = 0;
        let mut recommendations: Vec<String> = Vec::new();

        if tasks_completed_48h == 0 {
            signals += 1;
            recommendations.push("No tasks completed in 48h — consider reviewing blocked tasks or splitting large tasks.".to_string());
        }
        if avg_frustration > 0.6 {
            signals += 1;
            recommendations.push(format!(
                "High avg frustration ({:.2}) — consider abandoning stuck tasks or reassessing scope.",
                avg_frustration
            ));
        }
        if energy_trend < 0.0 {
            signals += 1;
            recommendations.push(format!(
                "Note energy declining (mean: {:.2}) — run deep_maintenance to consolidate knowledge.",
                mean_energy
            ));
        }
        if commits_48h == 0 {
            signals += 1;
            recommendations
                .push("No commits in 48h — project may be abandoned or blocked.".to_string());
        }

        let is_stagnating = signals >= 3;

        if is_stagnating {
            recommendations.push(
                "⚠️ Global stagnation detected — recommend running deep_maintenance.".to_string(),
            );
        }

        Ok(crate::neo4j::models::StagnationReport {
            is_stagnating,
            tasks_completed_48h,
            avg_frustration,
            energy_trend,
            commits_48h,
            signals_triggered: signals,
            recommendations,
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
    ///
    /// The `weak_synapse` boundary is derived from the actual weight distribution
    /// (p25 of all weights, fallback 0.3) instead of a hardcoded constant.
    pub async fn get_neural_metrics(&self, project_id: Uuid) -> Result<NeuralMetrics> {
        // Derive an adaptive weak-synapse threshold from the weight distribution.
        // Reuses get_all_synapse_weights from the note module — no duplicate query.
        // Falls back to 0.3 when fewer than 4 synapses exist.
        let weights = self
            .get_all_synapse_weights(Some(project_id))
            .await
            .unwrap_or_default();
        let weak_threshold =
            crate::analytics::distribution::adaptive_threshold(&weights, 0.25, 0.3);

        tracing::debug!(
            weak_threshold,
            n_synapses = weights.len(),
            "get_neural_metrics: adaptive weak-synapse threshold"
        );

        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (n:Note)-[:LINKED_TO]->(f)
            WITH count(DISTINCT n) AS total_notes
            OPTIONAL MATCH (n1:Note)-[s:SYNAPSE]->(n2:Note) WHERE s.weight >= 0.1
            WITH total_notes, count(s) AS active_synapses, avg(s.weight) AS avg_weight,
                 sum(CASE WHEN s.weight < $weak_threshold THEN 1 ELSE 0 END) AS weak_synapses
            OPTIONAL MATCH (n:Note) WHERE n.energy IS NOT NULL AND n.energy < 0.05
            WITH total_notes, active_synapses, coalesce(avg_weight, 0) AS avg_weight,
                 coalesce(weak_synapses, 0) AS weak_synapses, count(n) AS dead_notes
            RETURN active_synapses, avg_weight AS avg_energy,
                   CASE WHEN active_synapses > 0 THEN toFloat(weak_synapses) / active_synapses ELSE 0.0 END AS weak_synapses_ratio,
                   dead_notes AS dead_notes_count
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("weak_threshold", weak_threshold);

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
        // Simplified query: return raw values, normalization happens in Rust
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WITH f,
                 COALESCE(f.fabric_pagerank, f.pagerank, 0.0) AS pr,
                 COALESCE(f.churn_score, 0.0) AS churn,
                 COALESCE(f.knowledge_density, 0.0) AS density,
                 COALESCE(f.fabric_betweenness, f.betweenness, 0.0) AS bw
            RETURN f.path AS path, pr, churn, density, bw
            ORDER BY path
            "#,
        )
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Collect raw values for percentile computation
        struct RawFile {
            path: String,
            pr: f64,
            churn: f64,
            density: f64,
            bw: f64,
        }

        let raw_files: Vec<RawFile> = rows
            .iter()
            .filter_map(|row| {
                let path = row.get::<String>("path").ok()?;
                Some(RawFile {
                    path,
                    pr: row.get::<f64>("pr").unwrap_or(0.0),
                    churn: row.get::<f64>("churn").unwrap_or(0.0),
                    density: row.get::<f64>("density").unwrap_or(0.0),
                    bw: row.get::<f64>("bw").unwrap_or(0.0),
                })
            })
            .collect();

        if raw_files.is_empty() {
            return Ok(vec![]);
        }

        // Apply percentiles in Rust: log for pagerank (power-law), linear for betweenness
        let pr_vals: Vec<f64> = raw_files.iter().map(|f| f.pr).collect();
        let bw_vals: Vec<f64> = raw_files.iter().map(|f| f.bw).collect();

        let pr_pct = crate::graph::algorithms::to_log_percentiles(&pr_vals);
        let bw_pct = crate::graph::algorithms::to_linear_percentiles(&bw_vals);

        // Step 1 — compute raw risk scores for all files (formula unchanged)
        struct InterimScore {
            path: String,
            risk_score: f64,
            factors: RiskFactors,
        }

        let interim: Vec<InterimScore> = raw_files
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let pr_percentile = pr_pct[i];
                let bw_percentile = bw_pct[i];
                let risk_score = 0.3 * pr_percentile
                    + 0.3 * f.churn
                    + 0.25 * (1.0 - f.density)
                    + 0.15 * bw_percentile;

                InterimScore {
                    path: f.path.clone(),
                    risk_score,
                    factors: RiskFactors {
                        pagerank: pr_percentile,
                        churn: f.churn,
                        knowledge_gap: 1.0 - f.density,
                        betweenness: bw_percentile,
                    },
                }
            })
            .collect();

        // Step 2 — derive adaptive thresholds from the actual risk distribution
        // rather than hardcoded constants (0.75 / 0.5 / 0.25).
        // Fallbacks match the previous defaults so behaviour is unchanged
        // when n < 4 (adaptive_threshold returns fallback on empty/tiny slices).
        let risk_vals: Vec<f64> = interim.iter().map(|s| s.risk_score).collect();
        let critical_threshold =
            crate::analytics::distribution::adaptive_threshold(&risk_vals, 0.90, 0.75);
        let high_threshold =
            crate::analytics::distribution::adaptive_threshold(&risk_vals, 0.70, 0.50);
        let medium_threshold =
            crate::analytics::distribution::adaptive_threshold(&risk_vals, 0.40, 0.25);

        tracing::debug!(
            critical_threshold,
            high_threshold,
            medium_threshold,
            n = risk_vals.len(),
            "compute_risk_scores: adaptive thresholds derived from distribution"
        );

        // Step 3 — classify each file using the project-specific thresholds
        let mut scores: Vec<FileRiskScore> = interim
            .into_iter()
            .map(|s| {
                let risk_level = if s.risk_score >= critical_threshold {
                    "critical"
                } else if s.risk_score >= high_threshold {
                    "high"
                } else if s.risk_score >= medium_threshold {
                    "medium"
                } else {
                    "low"
                }
                .to_string();

                FileRiskScore {
                    path: s.path,
                    risk_score: s.risk_score,
                    risk_level,
                    factors: s.factors,
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            b.risk_score
                .partial_cmp(&a.risk_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

    // ========================================================================
    // Audit Gaps — Knowledge graph gap detection for system-inference protocol
    // ========================================================================

    /// Audit the knowledge graph for a project and return a structured gap report.
    /// Detects: orphan notes, decisions without AFFECTS, commits without TOUCHES,
    /// skills without members, and dynamic relationship type inventory.
    pub async fn audit_knowledge_gaps(&self, project_id: Uuid) -> Result<AuditGapsReport> {
        let pid = project_id.to_string();

        // 1. Notes without any LINKED_TO relations
        let orphan_notes_q = query(
            r#"
            MATCH (n:Note {project_id: $pid})
            WHERE NOT (n)-[:LINKED_TO]->()
            RETURN n.id AS id, left(n.content, 80) AS preview
            LIMIT 100
            "#,
        )
        .param("pid", pid.clone());

        let mut orphan_notes = Vec::new();
        let mut rows = self.graph.execute(orphan_notes_q).await?;
        while let Some(row) = rows.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let preview: String = row.get("preview").unwrap_or_default();
            orphan_notes.push(format!("{} — {}", &id[..8.min(id.len())], preview));
        }

        // 2. Decisions without AFFECTS relations
        let decisions_q = query(
            r#"
            MATCH (d:Decision)
            WHERE d.project_id = $pid
            AND NOT (d)-[:AFFECTS]->()
            RETURN d.id AS id, left(d.description, 80) AS preview
            LIMIT 100
            "#,
        )
        .param("pid", pid.clone());

        let mut decisions_without_affects = Vec::new();
        let mut rows = self.graph.execute(decisions_q).await?;
        while let Some(row) = rows.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let preview: String = row.get("preview").unwrap_or_default();
            decisions_without_affects.push(format!("{} — {}", &id[..8.min(id.len())], preview));
        }

        // 3. Commits without TOUCHES relations
        let commits_q = query(
            r#"
            MATCH (c:Commit {project_id: $pid})
            WHERE NOT (c)-[:TOUCHES]->()
            RETURN c.hash AS hash, left(c.message, 80) AS msg
            LIMIT 100
            "#,
        )
        .param("pid", pid.clone());

        let mut commits_without_touches = Vec::new();
        let mut rows = self.graph.execute(commits_q).await?;
        while let Some(row) = rows.next().await? {
            let hash: String = row.get("hash").unwrap_or_default();
            let msg: String = row.get("msg").unwrap_or_default();
            commits_without_touches.push(format!("{} — {}", &hash[..8.min(hash.len())], msg));
        }

        // 4. Skills without HAS_MEMBER relations
        let skills_q = query(
            r#"
            MATCH (s:Skill {project_id: $pid})
            WHERE NOT (s)-[:HAS_MEMBER]->()
            RETURN s.id AS id, s.name AS name
            LIMIT 50
            "#,
        )
        .param("pid", pid.clone());

        let mut skills_without_members = Vec::new();
        let mut rows = self.graph.execute(skills_q).await?;
        while let Some(row) = rows.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let name: String = row.get("name").unwrap_or_default();
            skills_without_members.push(format!("{} — {}", &id[..8.min(id.len())], name));
        }

        // 5. Dynamic relationship type inventory with actual counts
        let rel_types_q = query(
            r#"
            CALL db.relationshipTypes() YIELD relationshipType AS rel_type
            CALL {
                WITH rel_type
                MATCH ()-[r]->() WHERE type(r) = rel_type
                RETURN count(r) AS cnt
            }
            RETURN rel_type, cnt
            ORDER BY rel_type
            "#,
        );

        let mut relationship_type_counts = Vec::new();
        if let Ok(mut rows) = self.graph.execute(rel_types_q).await {
            while let Some(row) = rows.next().await? {
                let rel_type: String = row.get("rel_type").unwrap_or_default();
                let count: i64 = row.get("cnt").unwrap_or(0);
                relationship_type_counts.push(RelTypeCount { rel_type, count });
            }
        }

        let total_gaps = orphan_notes.len()
            + decisions_without_affects.len()
            + commits_without_touches.len()
            + skills_without_members.len();

        Ok(AuditGapsReport {
            total_gaps,
            orphan_notes,
            decisions_without_affects,
            commits_without_touches,
            skills_without_members,
            relationship_type_counts,
        })
    }

    // ========================================================================
    // Homeostasis — Bio-inspired auto-regulation metrics
    // ========================================================================

    /// Compute homeostasis report for a project's knowledge graph.
    /// Returns 5 ratios measuring the "health equilibrium" of the graph,
    /// inspired by biological homeostasis (Elun's homeostasis.rs).
    ///
    /// Default target ranges can be overridden via `custom_ranges`.
    pub async fn compute_homeostasis(
        &self,
        project_id: Uuid,
        custom_ranges: Option<&[(String, f64, f64)]>,
    ) -> Result<HomeostasisReport> {
        let pid = project_id.to_string();

        // Single Cypher query to collect all 5 ratios in one round-trip
        let q = query(
            r#"
            // 1. Note density: notes / files
            OPTIONAL MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WITH count(DISTINCT f) AS file_count
            OPTIONAL MATCH (n:Note {project_id: $pid})
            WHERE n.status = 'active'
            WITH file_count, count(DISTINCT n) AS note_count

            // 2. Decision coverage: decisions with AFFECTS / total files modified
            // Decisions don't have project_id — traverse Task→Plan→Project chain
            OPTIONAL MATCH (p2:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(:Task)-[:INFORMED_BY]->(d:Decision)-[:AFFECTS]->(target)
            WITH file_count, note_count,
                 count(DISTINCT target) AS files_with_decisions
            OPTIONAL MATCH (p3:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(:Task)-[:INFORMED_BY]->(d2:Decision)
            WITH file_count, note_count, files_with_decisions,
                 count(DISTINCT d2) AS total_decisions

            // 3. Synapse health: active synapses / active notes
            OPTIONAL MATCH (n1:Note {project_id: $pid})-[syn:SYNAPSE]->(n2:Note)
            WHERE syn.weight > 0.1
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 count(syn) AS active_synapses

            // 4. Hotspot coverage: hotspots with notes / total hotspots
            OPTIONAL MATCH (p:Project {id: $pid})-[:CONTAINS]->(hf:File)
            WHERE hf.churn_score IS NOT NULL AND hf.churn_score > 0.5
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 active_synapses, count(DISTINCT hf) AS hotspot_count
            OPTIONAL MATCH (p:Project {id: $pid})-[:CONTAINS]->(hf2:File)
            WHERE hf2.churn_score IS NOT NULL AND hf2.churn_score > 0.5
            AND EXISTS { MATCH (n:Note)-[:LINKED_TO]->(hf2) WHERE n.status = 'active' }
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 active_synapses, hotspot_count, count(DISTINCT hf2) AS covered_hotspots

            // 5. Scar load: scarred nodes / total notes+decisions
            OPTIONAL MATCH (sn:Note {project_id: $pid})
            WHERE sn.scar_intensity IS NOT NULL AND sn.scar_intensity > 0.0
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 active_synapses, hotspot_count, covered_hotspots,
                 count(sn) AS scarred_notes
            OPTIONAL MATCH (p5:Project {id: $pid})-[:HAS_PLAN]->(:Plan)-[:HAS_TASK]->(:Task)-[:INFORMED_BY]->(sd:Decision)
            WHERE sd.scar_intensity IS NOT NULL AND sd.scar_intensity > 0.0
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 active_synapses, hotspot_count, covered_hotspots,
                 scarred_notes, count(sd) AS scarred_decisions
            WITH file_count, note_count, files_with_decisions, total_decisions,
                 active_synapses, hotspot_count, covered_hotspots,
                 scarred_notes + scarred_decisions AS scarred_total

            RETURN file_count, note_count, files_with_decisions, total_decisions,
                   active_synapses, hotspot_count, covered_hotspots, scarred_total
            "#,
        )
        .param("pid", pid);

        let mut rows = self.graph.execute(q).await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| anyhow::anyhow!("No homeostasis data returned"))?;

        let file_count: i64 = row.get("file_count").unwrap_or(0);
        let note_count: i64 = row.get("note_count").unwrap_or(0);
        let files_with_decisions: i64 = row.get("files_with_decisions").unwrap_or(0);
        let total_decisions: i64 = row.get("total_decisions").unwrap_or(0);
        let active_synapses: i64 = row.get("active_synapses").unwrap_or(0);
        let hotspot_count: i64 = row.get("hotspot_count").unwrap_or(0);
        let covered_hotspots: i64 = row.get("covered_hotspots").unwrap_or(0);
        let scarred_total: i64 = row.get("scarred_total").unwrap_or(0);

        // Default target ranges (overridable via analysis_profile)
        let default_ranges: Vec<(&str, f64, f64)> = vec![
            ("note_density", 0.3, 2.0),      // 0.3-2.0 notes per file
            ("decision_coverage", 0.1, 0.8), // 10-80% of files have decisions
            ("synapse_health", 0.2, 3.0),    // 0.2-3.0 synapses per note
            ("churn_balance", 0.3, 1.0),     // 30-100% hotspots covered
            ("scar_load", 0.0, 0.15),        // 0-15% scarred nodes
        ];

        let get_range = |name: &str, default_min: f64, default_max: f64| -> (f64, f64) {
            if let Some(ranges) = custom_ranges {
                for (n, min, max) in ranges {
                    if n == name {
                        return (*min, *max);
                    }
                }
            }
            (default_min, default_max)
        };

        let mut ratios = Vec::new();
        let mut total_pain = 0.0;

        // 1. Note density
        let note_density = if file_count > 0 {
            note_count as f64 / file_count as f64
        } else {
            0.0
        };
        let (min, max) = get_range("note_density", default_ranges[0].1, default_ranges[0].2);
        ratios.push(Self::make_ratio("note_density", note_density, min, max));

        // 2. Decision coverage
        let decision_cov = if file_count > 0 {
            files_with_decisions as f64 / file_count as f64
        } else {
            0.0
        };
        let (min, max) = get_range(
            "decision_coverage",
            default_ranges[1].1,
            default_ranges[1].2,
        );
        ratios.push(Self::make_ratio(
            "decision_coverage",
            decision_cov,
            min,
            max,
        ));

        // 3. Synapse health
        let synapse_ratio = if note_count > 0 {
            active_synapses as f64 / note_count as f64
        } else {
            0.0
        };
        let (min, max) = get_range("synapse_health", default_ranges[2].1, default_ranges[2].2);
        ratios.push(Self::make_ratio("synapse_health", synapse_ratio, min, max));

        // 4. Churn balance
        let churn_bal = if hotspot_count > 0 {
            covered_hotspots as f64 / hotspot_count as f64
        } else {
            1.0 // no hotspots = perfectly balanced
        };
        let (min, max) = get_range("churn_balance", default_ranges[3].1, default_ranges[3].2);
        ratios.push(Self::make_ratio("churn_balance", churn_bal, min, max));

        // 5. Scar load
        let total_nodes = (note_count + total_decisions).max(1) as f64;
        let scar_ratio = scarred_total as f64 / total_nodes;
        let (min, max) = get_range("scar_load", default_ranges[4].1, default_ranges[4].2);
        ratios.push(Self::make_ratio("scar_load", scar_ratio, min, max));

        // Aggregate pain score (mean of normalized distances, clamped to [0, 1])
        for r in &ratios {
            total_pain += r.distance_to_equilibrium;
        }
        let pain_score = (total_pain / ratios.len() as f64).clamp(0.0, 1.0);

        // Generate overall recommendations
        let mut recommendations = Vec::new();
        for r in &ratios {
            if let Some(ref rec) = r.recommendation {
                recommendations.push(rec.clone());
            }
        }

        Ok(HomeostasisReport {
            ratios,
            pain_score,
            recommendations,
        })
    }

    /// Helper: build a HomeostasisRatio with distance and severity computation.
    fn make_ratio(name: &str, value: f64, min: f64, max: f64) -> HomeostasisRatio {
        let distance = if value < min {
            (min - value) / min.max(0.01)
        } else if value > max {
            (value - max) / max.max(0.01)
        } else {
            0.0
        };
        let distance = distance.clamp(0.0, 2.0); // cap at 2.0

        let severity = if distance == 0.0 {
            HomeostasisSeverity::Ok
        } else if distance < 0.5 {
            HomeostasisSeverity::Warning
        } else {
            HomeostasisSeverity::Critical
        };

        let recommendation = match (name, &severity) {
            (_, HomeostasisSeverity::Ok) => None,
            ("note_density", _) if value < min => {
                Some(format!("Knowledge gap: only {:.2} notes/file (target ≥ {:.1}). Add notes to under-documented files.", value, min))
            }
            ("note_density", _) => {
                Some(format!("Over-documentation: {:.2} notes/file (target ≤ {:.1}). Consider consolidating redundant notes.", value, max))
            }
            ("decision_coverage", _) if value < min => {
                Some(format!("Low decision coverage: {:.0}% of files have architectural decisions (target ≥ {:.0}%). Add AFFECTS links to decisions.", value * 100.0, min * 100.0))
            }
            ("decision_coverage", _) => {
                Some(format!("High decision coverage: {:.0}% (target ≤ {:.0}%). Some decisions may be too granular.", value * 100.0, max * 100.0))
            }
            ("synapse_health", _) if value < min => {
                Some(format!("Weak neural network: {:.2} synapses/note (target ≥ {:.1}). Run reinforce_neurons on related notes.", value, min))
            }
            ("synapse_health", _) => {
                Some(format!("Dense neural network: {:.2} synapses/note (target ≤ {:.1}). Run decay_synapses to prune weak links.", value, max))
            }
            ("churn_balance", _) if value < min => {
                Some(format!("Hotspot blind spots: only {:.0}% of frequently-changed files have notes (target ≥ {:.0}%). Prioritize documenting hotspots.", value * 100.0, min * 100.0))
            }
            ("churn_balance", _) => None, // can't have too much coverage
            ("scar_load", _) => {
                Some(format!("High scar load: {:.1}% of nodes are scarred (target ≤ {:.0}%). Run heal_scars on resolved issues.", value * 100.0, max * 100.0))
            }
            _ => None,
        };

        HomeostasisRatio {
            name: name.to_string(),
            value,
            target_range: (min, max),
            distance_to_equilibrium: distance,
            severity,
            recommendation,
        }
    }

    // ========================================================================
    // Identity Manifold — Community centroids & structural drift
    // ========================================================================

    /// Compute structural drift for all files in a project.
    ///
    /// 1. Fetches all files with fingerprint + community_id from Neo4j (single query)
    /// 2. Groups by community, computes centroid (mean fingerprint) per community
    /// 3. Computes euclidean distance from each file to its community centroid
    /// 4. Returns files sorted by drift (descending) with severity classification
    ///
    /// Thresholds (configurable via `warning_threshold` and `critical_threshold`):
    /// - ok: distance < warning_threshold
    /// - warning: warning_threshold ≤ distance < critical_threshold
    /// - critical: distance ≥ critical_threshold
    pub async fn compute_structural_drift(
        &self,
        project_id: Uuid,
        warning_threshold: Option<f64>,
        critical_threshold: Option<f64>,
    ) -> Result<crate::neo4j::models::StructuralDriftReport> {
        use crate::neo4j::models::{
            CommunityIdentity, HomeostasisSeverity, StructuralDrift, StructuralDriftReport,
        };

        let warn_thresh = warning_threshold.unwrap_or(1.5);
        let crit_thresh = critical_threshold.unwrap_or(3.0);

        // Single query: get all files with fingerprint + community data
        let q = query(
            r#"
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(f:File)
            WHERE f.structural_fingerprint IS NOT NULL
              AND f.community_id IS NOT NULL
            RETURN f.path AS path,
                   f.structural_fingerprint AS fingerprint,
                   f.community_id AS community_id,
                   COALESCE(f.community_label, 'Community ' + toString(f.community_id)) AS community_label
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;

        // Collect all files
        struct FileEntry {
            path: String,
            fingerprint: Vec<f64>,
            community_id: i64,
            community_label: String,
        }

        let mut files: Vec<FileEntry> = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(path), Ok(fp), Ok(cid)) = (
                row.get::<String>("path"),
                row.get::<Vec<f64>>("fingerprint"),
                row.get::<i64>("community_id"),
            ) {
                let label = row
                    .get::<String>("community_label")
                    .unwrap_or_else(|_| format!("Community {}", cid));
                files.push(FileEntry {
                    path,
                    fingerprint: fp,
                    community_id: cid,
                    community_label: label,
                });
            }
        }

        if files.is_empty() {
            return Ok(StructuralDriftReport {
                drifting_files: vec![],
                centroids: vec![],
                mean_drift: 0.0,
                warning_count: 0,
                critical_count: 0,
            });
        }

        // Group by community and compute centroids
        let mut community_files: std::collections::HashMap<i64, Vec<&FileEntry>> =
            std::collections::HashMap::new();
        for file in &files {
            community_files
                .entry(file.community_id)
                .or_default()
                .push(file);
        }

        let now = chrono::Utc::now();
        let mut centroids: Vec<CommunityIdentity> = Vec::new();
        let mut centroid_map: std::collections::HashMap<i64, Vec<f64>> =
            std::collections::HashMap::new();

        for (cid, members) in &community_files {
            let n = members.len();
            if n == 0 {
                continue;
            }

            // Determine dimensionality from first member
            let dims = members[0].fingerprint.len();
            let mut centroid = vec![0.0f64; dims];

            for m in members {
                for (i, &v) in m.fingerprint.iter().enumerate() {
                    if i < dims {
                        centroid[i] += v;
                    }
                }
            }
            for v in centroid.iter_mut() {
                *v /= n as f64;
            }

            centroids.push(CommunityIdentity {
                community_id: *cid,
                community_label: members[0].community_label.clone(),
                centroid: centroid.clone(),
                member_count: n,
                last_computed: now,
            });
            centroid_map.insert(*cid, centroid);
        }

        // Compute drift for each file
        let mut drifting_files: Vec<StructuralDrift> = Vec::new();
        let mut total_drift = 0.0;

        for file in &files {
            if let Some(centroid) = centroid_map.get(&file.community_id) {
                let distance = euclidean_distance(&file.fingerprint, centroid);
                let severity = if distance >= crit_thresh {
                    HomeostasisSeverity::Critical
                } else if distance >= warn_thresh {
                    HomeostasisSeverity::Warning
                } else {
                    HomeostasisSeverity::Ok
                };

                let suggestion = match severity {
                    HomeostasisSeverity::Critical => Some(format!(
                        "File has drifted significantly from Community {} identity (distance: {:.2}). Consider migrating to a structurally closer community.",
                        file.community_id, distance
                    )),
                    HomeostasisSeverity::Warning => Some(format!(
                        "Moderate structural drift from Community {} (distance: {:.2}). Monitor for further divergence.",
                        file.community_id, distance
                    )),
                    HomeostasisSeverity::Ok => None,
                };

                total_drift += distance;

                drifting_files.push(StructuralDrift {
                    file_path: file.path.clone(),
                    community_id: file.community_id,
                    community_label: file.community_label.clone(),
                    drift_distance: distance,
                    severity,
                    suggestion,
                });
            }
        }

        // Sort by drift descending
        drifting_files.sort_by(|a, b| {
            b.drift_distance
                .partial_cmp(&a.drift_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mean_drift = if files.is_empty() {
            0.0
        } else {
            total_drift / files.len() as f64
        };

        let warning_count = drifting_files
            .iter()
            .filter(|f| matches!(f.severity, HomeostasisSeverity::Warning))
            .count();
        let critical_count = drifting_files
            .iter()
            .filter(|f| matches!(f.severity, HomeostasisSeverity::Critical))
            .count();

        Ok(StructuralDriftReport {
            drifting_files,
            centroids,
            mean_drift,
            warning_count,
            critical_count,
        })
    }
    // =========================================================================
    // rs-stats data providers — bulk metric retrieval for statistical analysis
    // =========================================================================

    /// Fetch all PageRank values for a project as a flat vector.
    ///
    /// Returns an empty Vec if no analytics have been computed yet.
    /// Used by `analytics::distribution::analyze_distribution` to produce a
    /// full distribution analysis (best-fit model, p95 threshold, etc.).
    pub async fn get_all_pagerank_values(&self, project_id: Uuid) -> Result<Vec<f64>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(n)
            WHERE (n:File OR n:Function) AND n.pagerank IS NOT NULL
            RETURN toFloat(n.pagerank) AS pr
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get::<f64>("pr").ok())
            .collect())
    }

    /// Fetch risk scores grouped by community for ANOVA analysis.
    ///
    /// Returns a Vec of Vecs — each inner Vec is the risk scores of one community.
    /// Communities with fewer than 2 files are excluded (ANOVA precondition).
    /// Returns an empty Vec if risk scores have not been computed yet.
    pub async fn get_community_risk_vectors(&self, project_id: Uuid) -> Result<Vec<Vec<f64>>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.risk_score IS NOT NULL AND f.community_id IS NOT NULL
            WITH f.community_id AS cid, collect(toFloat(f.risk_score)) AS scores
            WHERE size(scores) >= 2
            RETURN scores
            ORDER BY cid
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        let groups = rows
            .iter()
            .filter_map(|r| r.get::<Vec<f64>>("scores").ok())
            .filter(|g| g.len() >= 2)
            .collect();
        Ok(groups)
    }

    /// Fetch all risk scores for a project as a flat vector.
    ///
    /// Returns an empty Vec if risk scores have not been computed yet.
    pub async fn get_all_risk_score_values(&self, project_id: Uuid) -> Result<Vec<f64>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File)
            WHERE f.risk_score IS NOT NULL
            RETURN toFloat(f.risk_score) AS rs
            "#,
        )
        .param("pid", project_id.to_string());

        let rows = self.execute_with_params(q).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get::<f64>("rs").ok())
            .collect())
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
