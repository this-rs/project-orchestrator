//! Neo4j Topology Firewall operations (GraIL Plan 3)
//!
//! CRUD for topology rules and violation checking.
//! Inspired by GraIL negative triples — encodes architectural constraints
//! as forbidden relationships in the knowledge graph.

use super::client::Neo4jClient;
use crate::graph::models::{
    glob_to_regex, TopologyRule, TopologyRuleType, TopologySeverity, TopologyViolation,
};
use anyhow::Result;
use neo4rs::query;
use tracing::warn;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to a [`TopologyRule`].
    fn node_to_topology_rule(&self, node: &neo4rs::Node) -> Result<TopologyRule> {
        let rule_type_str: String = node.get("rule_type")?;
        let rule_type = TopologyRuleType::from_str_loose(&rule_type_str)
            .ok_or_else(|| anyhow::anyhow!("Unknown rule_type: {}", rule_type_str))?;

        let severity_str: String = node.get("severity")?;
        let severity = TopologySeverity::from_str_loose(&severity_str)
            .ok_or_else(|| anyhow::anyhow!("Unknown severity: {}", severity_str))?;

        let threshold: Option<i64> = node.get::<i64>("threshold").ok();
        let target_pattern: Option<String> = node
            .get::<String>("target_pattern")
            .ok()
            .filter(|s| !s.is_empty());

        Ok(TopologyRule {
            id: node.get("id")?,
            project_id: node.get("project_id")?,
            rule_type,
            source_pattern: node.get("source_pattern")?,
            target_pattern,
            threshold: threshold.map(|t| t as u32),
            severity,
            description: node.get("description")?,
        })
    }

    // ========================================================================
    // CRUD operations
    // ========================================================================

    /// Create a topology rule.
    pub async fn create_topology_rule(&self, rule: &TopologyRule) -> Result<()> {
        let q = query(
            r#"
            CREATE (r:TopologyRule {
                id: $id,
                project_id: $project_id,
                rule_type: $rule_type,
                source_pattern: $source_pattern,
                target_pattern: $target_pattern,
                threshold: $threshold,
                severity: $severity,
                description: $description
            })
            "#,
        )
        .param("id", rule.id.clone())
        .param("project_id", rule.project_id.clone())
        .param("rule_type", rule.rule_type.to_string())
        .param("source_pattern", rule.source_pattern.clone())
        .param(
            "target_pattern",
            rule.target_pattern.clone().unwrap_or_default(),
        )
        .param("threshold", rule.threshold.unwrap_or(0) as i64)
        .param("severity", rule.severity.to_string())
        .param("description", rule.description.clone());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List all topology rules for a project.
    pub async fn list_topology_rules(&self, project_id: &str) -> Result<Vec<TopologyRule>> {
        let q = query(
            r#"
            MATCH (r:TopologyRule {project_id: $project_id})
            RETURN r
            ORDER BY r.rule_type, r.source_pattern
            "#,
        )
        .param("project_id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut rules = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("r")?;
            match self.node_to_topology_rule(&node) {
                Ok(rule) => rules.push(rule),
                Err(e) => warn!(error = %e, "Failed to parse TopologyRule node"),
            }
        }
        Ok(rules)
    }

    /// Delete a topology rule by id.
    pub async fn delete_topology_rule(&self, rule_id: &str) -> Result<()> {
        let q = query(
            r#"
            MATCH (r:TopologyRule {id: $id})
            DETACH DELETE r
            "#,
        )
        .param("id", rule_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Violation checking
    // ========================================================================

    /// Check all topology rules for a project and return violations.
    ///
    /// Dispatches to rule-type-specific Cypher queries.
    /// All queries are scoped by project_id.
    pub async fn check_topology_rules(
        &self,
        project_id: &str,
    ) -> Result<Vec<TopologyViolation>> {
        let rules = self.list_topology_rules(project_id).await?;
        let mut all_violations = Vec::new();

        for rule in &rules {
            let violations = match rule.rule_type {
                TopologyRuleType::MustNotImport => {
                    self.check_must_not_traverse(rule, project_id, "IMPORTS")
                        .await?
                }
                TopologyRuleType::MustNotCall => {
                    self.check_must_not_traverse(rule, project_id, "CALLS")
                        .await?
                }
                TopologyRuleType::MaxFanOut => {
                    self.check_max_fan_out(rule, project_id).await?
                }
                TopologyRuleType::NoCircular => {
                    self.check_no_circular(rule, project_id).await?
                }
                TopologyRuleType::MaxDistance => {
                    self.check_max_distance(rule, project_id).await?
                }
            };
            all_violations.extend(violations);
        }

        // Sort by violation_score descending (most dangerous first)
        all_violations.sort_by(|a, b| {
            b.violation_score
                .partial_cmp(&a.violation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_violations)
    }

    /// Check MUST_NOT_IMPORT / MUST_NOT_CALL violations.
    ///
    /// Finds files/functions matching source_pattern that have a relationship
    /// of the given type to files/functions matching target_pattern.
    async fn check_must_not_traverse(
        &self,
        rule: &TopologyRule,
        project_id: &str,
        rel_type: &str,
    ) -> Result<Vec<TopologyViolation>> {
        let source_regex = glob_to_regex(&rule.source_pattern);
        let target_pattern = rule.target_pattern.as_deref().unwrap_or("**");
        let target_regex = glob_to_regex(target_pattern);

        // Whitelist the relationship type to prevent injection
        let safe_rel_type = match rel_type {
            "IMPORTS" => "IMPORTS",
            "CALLS" => "CALLS",
            _ => return Ok(Vec::new()),
        };

        // We use two separate queries for IMPORTS and CALLS since
        // dynamic relationship types in Cypher require APOC or string interpolation.
        // Since we whitelist above, this is safe.
        let cypher = if safe_rel_type == "IMPORTS" {
            r#"
            MATCH (s:File {project_id: $project_id})-[:IMPORTS]->(t:File {project_id: $project_id})
            WHERE s.path =~ $source_regex AND t.path =~ $target_regex
            RETURN s.path AS violator, t.path AS target
            LIMIT 100
            "#
        } else {
            r#"
            MATCH (s:Function {project_id: $project_id})-[:CALLS]->(t:Function {project_id: $project_id})
            WHERE s.path =~ $source_regex AND t.path =~ $target_regex
            RETURN s.path AS violator, t.path AS target
            LIMIT 100
            "#
        };

        let q = query(cypher)
            .param("project_id", project_id.to_string())
            .param("source_regex", source_regex)
            .param("target_regex", target_regex);

        let mut result = self.graph.execute(q).await?;
        let mut violations = Vec::new();

        while let Some(row) = result.next().await? {
            let violator: String = row.get("violator")?;
            let target: String = row.get("target")?;

            // Compute violation_score: for forbidden edges, structural plausibility
            // is approximated by the fact that the edge exists (= 1.0).
            // severity_weight: error = 1.0, warning = 0.5
            let severity_weight = match rule.severity {
                TopologySeverity::Error => 1.0,
                TopologySeverity::Warning => 0.5,
            };
            let violation_score = severity_weight; // existing edge → plausibility = 1.0

            violations.push(TopologyViolation {
                rule_id: rule.id.clone(),
                rule_description: rule.description.clone(),
                rule_type: rule.rule_type.clone(),
                violator_path: violator.clone(),
                target_path: Some(target.clone()),
                severity: rule.severity.clone(),
                details: format!(
                    "{} {} {} (forbidden by rule)",
                    violator, safe_rel_type, target
                ),
                violation_score,
            });
        }

        Ok(violations)
    }

    /// Check MAX_FAN_OUT violations.
    ///
    /// Counts outgoing IMPORTS for each file matching source_pattern
    /// and flags those exceeding the threshold.
    async fn check_max_fan_out(
        &self,
        rule: &TopologyRule,
        project_id: &str,
    ) -> Result<Vec<TopologyViolation>> {
        let source_regex = glob_to_regex(&rule.source_pattern);
        let threshold = rule.threshold.unwrap_or(20) as i64;

        let q = query(
            r#"
            MATCH (f:File {project_id: $project_id})
            WHERE f.path =~ $source_regex
            WITH f, size([(f)-[:IMPORTS]->(x:File) | x]) AS fan_out
            WHERE fan_out > $threshold
            RETURN f.path AS violator, fan_out
            ORDER BY fan_out DESC
            LIMIT 100
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("source_regex", source_regex)
        .param("threshold", threshold);

        let mut result = self.graph.execute(q).await?;
        let mut violations = Vec::new();

        while let Some(row) = result.next().await? {
            let violator: String = row.get("violator")?;
            let fan_out: i64 = row.get("fan_out")?;

            // Score: how much over the limit (normalized)
            let overshoot = (fan_out - threshold) as f64 / threshold as f64;
            let severity_weight = match rule.severity {
                TopologySeverity::Error => 1.0,
                TopologySeverity::Warning => 0.5,
            };
            let violation_score = (overshoot.min(2.0) / 2.0) * severity_weight;

            violations.push(TopologyViolation {
                rule_id: rule.id.clone(),
                rule_description: rule.description.clone(),
                rule_type: rule.rule_type.clone(),
                violator_path: violator.clone(),
                target_path: None,
                severity: rule.severity.clone(),
                details: format!(
                    "{} has {} imports (threshold: {})",
                    violator, fan_out, threshold
                ),
                violation_score,
            });
        }

        Ok(violations)
    }

    /// Check NO_CIRCULAR violations.
    ///
    /// Detects circular import chains (depth 2..6) among files matching
    /// source_pattern. Limited to 10 results for performance.
    async fn check_no_circular(
        &self,
        rule: &TopologyRule,
        project_id: &str,
    ) -> Result<Vec<TopologyViolation>> {
        let source_regex = glob_to_regex(&rule.source_pattern);

        let q = query(
            r#"
            MATCH path = (f:File {project_id: $project_id})-[:IMPORTS*2..6]->(f)
            WHERE f.path =~ $source_regex
            WITH f, length(path) AS cycle_len
            RETURN DISTINCT f.path AS violator, min(cycle_len) AS shortest_cycle
            ORDER BY shortest_cycle ASC
            LIMIT 10
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("source_regex", source_regex);

        let mut result = self.graph.execute(q).await?;
        let mut violations = Vec::new();

        while let Some(row) = result.next().await? {
            let violator: String = row.get("violator")?;
            let shortest_cycle: i64 = row.get("shortest_cycle")?;

            // Shorter cycles are more dangerous
            let severity_weight = match rule.severity {
                TopologySeverity::Error => 1.0,
                TopologySeverity::Warning => 0.5,
            };
            let cycle_danger = 1.0 / shortest_cycle as f64; // cycle of 2 → 0.5, cycle of 6 → 0.17
            let violation_score = cycle_danger * severity_weight;

            violations.push(TopologyViolation {
                rule_id: rule.id.clone(),
                rule_description: rule.description.clone(),
                rule_type: rule.rule_type.clone(),
                violator_path: violator.clone(),
                target_path: None,
                severity: rule.severity.clone(),
                details: format!(
                    "{} is part of a circular import chain (shortest cycle: {} hops)",
                    violator, shortest_cycle
                ),
                violation_score,
            });
        }

        Ok(violations)
    }

    /// Check MAX_DISTANCE violations.
    ///
    /// Verifies that the shortest path between any file matching source_pattern
    /// and any file matching target_pattern is >= threshold.
    /// Violations are pairs where the distance is < threshold.
    async fn check_max_distance(
        &self,
        rule: &TopologyRule,
        project_id: &str,
    ) -> Result<Vec<TopologyViolation>> {
        let source_regex = glob_to_regex(&rule.source_pattern);
        let target_pattern = rule.target_pattern.as_deref().unwrap_or("**");
        let target_regex = glob_to_regex(target_pattern);
        let threshold = rule.threshold.unwrap_or(2) as i64;

        let q = query(
            r#"
            MATCH (s:File {project_id: $project_id}), (t:File {project_id: $project_id})
            WHERE s.path =~ $source_regex AND t.path =~ $target_regex AND s <> t
            WITH s, t
            MATCH p = shortestPath((s)-[:IMPORTS*..10]->(t))
            WITH s.path AS source_path, t.path AS target_path, length(p) AS dist
            WHERE dist < $threshold
            RETURN source_path, target_path, dist
            ORDER BY dist ASC
            LIMIT 50
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("source_regex", source_regex)
        .param("target_regex", target_regex)
        .param("threshold", threshold);

        let mut result = self.graph.execute(q).await?;
        let mut violations = Vec::new();

        while let Some(row) = result.next().await? {
            let source_path: String = row.get("source_path")?;
            let target_path: String = row.get("target_path")?;
            let dist: i64 = row.get("dist")?;

            let severity_weight = match rule.severity {
                TopologySeverity::Error => 1.0,
                TopologySeverity::Warning => 0.5,
            };
            // Closer = more dangerous
            let closeness = 1.0 - (dist as f64 / threshold as f64);
            let violation_score = closeness.max(0.0) * severity_weight;

            violations.push(TopologyViolation {
                rule_id: rule.id.clone(),
                rule_description: rule.description.clone(),
                rule_type: rule.rule_type.clone(),
                violator_path: source_path.clone(),
                target_path: Some(target_path.clone()),
                severity: rule.severity.clone(),
                details: format!(
                    "Distance from {} to {} is {} (minimum required: {})",
                    source_path, target_path, dist, threshold
                ),
                violation_score,
            });
        }

        Ok(violations)
    }

    /// Check if a specific file's new imports would violate any MustNotImport rules.
    ///
    /// **Performance target: <50ms** — Only fetches rules from Neo4j (1 query),
    /// then checks patterns in-memory using Rust regex. No Cypher graph traversal.
    ///
    /// Returns violations for each new_import that matches a forbidden target pattern
    /// where file_path matches the source pattern of a MustNotImport rule.
    pub async fn check_file_topology(
        &self,
        project_id: &str,
        file_path: &str,
        new_imports: &[String],
    ) -> Result<Vec<TopologyViolation>> {
        if new_imports.is_empty() {
            return Ok(vec![]);
        }

        // 1. Fetch MustNotImport and MustNotCall rules for this project (single query)
        let rules = self.list_topology_rules(project_id).await?;
        let relevant_rules: Vec<&TopologyRule> = rules
            .iter()
            .filter(|r| {
                matches!(
                    r.rule_type,
                    TopologyRuleType::MustNotImport | TopologyRuleType::MustNotCall
                )
            })
            .collect();

        if relevant_rules.is_empty() {
            return Ok(vec![]);
        }

        // 2. Check each rule in-memory using regex
        let mut violations = Vec::new();

        for rule in &relevant_rules {
            let source_regex_str = glob_to_regex(&rule.source_pattern);
            let source_re = match regex::Regex::new(&source_regex_str) {
                Ok(re) => re,
                Err(e) => {
                    warn!("Invalid source regex for rule {}: {}", rule.id, e);
                    continue;
                }
            };

            // Does file_path match the source pattern?
            if !source_re.is_match(file_path) {
                continue;
            }

            // Build target regex
            let target_pattern = rule.target_pattern.as_deref().unwrap_or("**");
            let target_regex_str = glob_to_regex(target_pattern);
            let target_re = match regex::Regex::new(&target_regex_str) {
                Ok(re) => re,
                Err(e) => {
                    warn!("Invalid target regex for rule {}: {}", rule.id, e);
                    continue;
                }
            };

            // Check each new import against the target pattern
            for import_path in new_imports {
                if target_re.is_match(import_path) {
                    let severity_weight = match rule.severity {
                        TopologySeverity::Error => 1.0,
                        TopologySeverity::Warning => 0.5,
                    };

                    violations.push(TopologyViolation {
                        rule_id: rule.id.clone(),
                        rule_description: rule.description.clone(),
                        rule_type: rule.rule_type.clone(),
                        violator_path: file_path.to_string(),
                        target_path: Some(import_path.clone()),
                        severity: rule.severity.clone(),
                        details: format!(
                            "{} would import {} which violates rule: {}",
                            file_path, import_path, rule.description
                        ),
                        violation_score: severity_weight,
                    });
                }
            }
        }

        // Sort by score descending
        violations.sort_by(|a, b| {
            b.violation_score
                .partial_cmp(&a.violation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(violations)
    }
}
