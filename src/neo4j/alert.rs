//! Neo4j Alert operations (Heartbeat Engine)

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use chrono::Utc;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Alert operations
    // ========================================================================

    /// Upsert an alert, collapsing repeat observations of the same condition.
    ///
    /// Keyed on `dedup_key`, NOT on a fresh uuid. A condition seen again
    /// increments `occurrence_count`, refreshes `last_seen` and `message`
    /// (latest wording wins) and recomputes `priority`. `acknowledged` is
    /// deliberately preserved on re-observation so acknowledging a known,
    /// ongoing condition stays acknowledged instead of resurfacing every tick.
    ///
    /// This replaces an append-only `CREATE`, which had produced 1,355,752
    /// `git_drift` nodes for 1,591 distinct messages — 40% of all nodes in the
    /// graph — because the commit count was embedded in the message.
    pub async fn create_alert_node(&self, alert: &AlertNode) -> Result<()> {
        let now = Utc::now();
        let dedup_key = if alert.dedup_key.is_empty() {
            AlertNode::make_dedup_key(&alert.alert_type, alert.project_id, &alert.message)
        } else {
            alert.dedup_key.clone()
        };

        let q = query(
            r#"
            MERGE (a:Alert {dedup_key: $dedup_key})
            ON CREATE SET
                a.id = $id,
                a.alert_type = $alert_type,
                a.severity = $severity,
                a.message = $message,
                a.project_id = $project_id,
                a.acknowledged = false,
                a.created_at = $now,
                a.first_seen = $now,
                a.occurrence_count = 0
            SET a.severity   = $severity,
                a.message    = $message,
                a.last_seen  = $now,
                a.occurrence_count = coalesce(a.occurrence_count, 0) + 1
            WITH a
            SET a.priority = CASE
                    WHEN coalesce(a.acknowledged, false) THEN 0.0
                    ELSE $sev_weight * (1.0 + (
                        CASE WHEN log(toFloat(a.occurrence_count)) / 10.0 > 0.5
                             THEN 0.5
                             ELSE log(toFloat(a.occurrence_count)) / 10.0 END))
                END
            WITH a
            OPTIONAL MATCH (p:Project {id: $project_id})
            FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
                MERGE (p)-[:HAS_ALERT]->(a)
            )
            "#,
        )
        .param("dedup_key", dedup_key)
        .param("id", alert.id.to_string())
        .param("alert_type", alert.alert_type.clone())
        .param("severity", alert.severity.to_string())
        .param("message", alert.message.clone())
        .param(
            "project_id",
            alert
                .project_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
        )
        .param("now", now.to_rfc3339())
        .param("sev_weight", alert.severity.weight());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List pending (unacknowledged) alerts, optionally filtered by project.
    pub async fn list_pending_alerts_impl(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
    ) -> Result<Vec<AlertNode>> {
        let cypher = if project_id.is_some() {
            r#"
            MATCH (a:Alert)
            WHERE a.acknowledged = false AND a.project_id = $project_id
            RETURN a
            ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC
            LIMIT $limit
            "#
        } else {
            r#"
            MATCH (a:Alert)
            WHERE a.acknowledged = false
            RETURN a
            ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC
            LIMIT $limit
            "#
        };

        let mut q = query(cypher).param("limit", limit as i64);
        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut alerts = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(alert) = Self::parse_alert_row(&row, "a") {
                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    /// Acknowledge an alert.
    pub async fn acknowledge_alert_impl(
        &self,
        alert_id: Uuid,
        acknowledged_by: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (a:Alert {id: $id})
            SET a.acknowledged = true,
                a.acknowledged_by = $acknowledged_by,
                a.acknowledged_at = $acknowledged_at
            "#,
        )
        .param("id", alert_id.to_string())
        .param("acknowledged_by", acknowledged_by.to_string())
        .param("acknowledged_at", Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a single alert by ID.
    pub async fn get_alert_impl(&self, alert_id: Uuid) -> Result<Option<AlertNode>> {
        let q = query(
            r#"
            MATCH (a:Alert {id: $id})
            RETURN a
            "#,
        )
        .param("id", alert_id.to_string());

        let mut result = self.graph.execute(q).await?;

        if let Some(row) = result.next().await? {
            Ok(Some(Self::parse_alert_row(&row, "a")?))
        } else {
            Ok(None)
        }
    }

    /// List all alerts with pagination.
    pub async fn list_alerts_impl(
        &self,
        project_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<AlertNode>, usize)> {
        // Count total
        let count_cypher = if project_id.is_some() {
            "MATCH (a:Alert) WHERE a.project_id = $project_id RETURN count(a) AS total"
        } else {
            "MATCH (a:Alert) RETURN count(a) AS total"
        };

        let mut count_q = query(count_cypher);
        if let Some(pid) = project_id {
            count_q = count_q.param("project_id", pid.to_string());
        }

        let mut count_result = self.graph.execute(count_q).await?;
        let total = if let Some(row) = count_result.next().await? {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        // Fetch page
        let list_cypher = if project_id.is_some() {
            r#"
            MATCH (a:Alert)
            WHERE a.project_id = $project_id
            RETURN a
            ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC
            SKIP $offset LIMIT $limit
            "#
        } else {
            r#"
            MATCH (a:Alert)
            RETURN a
            ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC
            SKIP $offset LIMIT $limit
            "#
        };

        let mut list_q = query(list_cypher)
            .param("limit", limit as i64)
            .param("offset", offset as i64);
        if let Some(pid) = project_id {
            list_q = list_q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(list_q).await?;
        let mut alerts = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(alert) = Self::parse_alert_row(&row, "a") {
                alerts.push(alert);
            }
        }

        Ok((alerts, total))
    }

    /// Parse an AlertNode from a Neo4j row.
    fn parse_alert_row(row: &neo4rs::Row, key: &str) -> Result<AlertNode> {
        let node: neo4rs::Node = row.get(key)?;

        let severity_str: String = node.get("severity")?;
        let severity: AlertSeverity = severity_str
            .parse()
            .map_err(|e: String| anyhow::anyhow!(e))?;

        let project_id = node
            .get::<String>("project_id")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse().ok());

        let acknowledged = node.get::<bool>("acknowledged").unwrap_or(false);

        let acknowledged_by = node
            .get::<String>("acknowledged_by")
            .ok()
            .filter(|s| !s.is_empty());

        let acknowledged_at = node
            .get::<String>("acknowledged_at")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let created_at = node
            .get::<String>("created_at")
            .ok()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);

        let parse_ts = |k: &str| -> Option<chrono::DateTime<chrono::Utc>> {
            node.get::<String>(k)
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc))
        };
        let first_seen = parse_ts("first_seen").or(Some(created_at));
        let last_seen = parse_ts("last_seen").or(Some(created_at));
        let occurrence_count = node.get::<i64>("occurrence_count").unwrap_or(1).max(1) as u64;
        let dedup_key = node.get::<String>("dedup_key").unwrap_or_default();
        let priority = node.get::<f64>("priority").unwrap_or_else(|_| {
            AlertNode::compute_priority(
                severity,
                occurrence_count,
                last_seen.unwrap_or(created_at),
                acknowledged,
                chrono::Utc::now(),
            )
        });

        Ok(AlertNode {
            id: node.get::<String>("id")?.parse()?,
            alert_type: node.get("alert_type")?,
            severity,
            message: node.get("message")?,
            project_id,
            acknowledged,
            acknowledged_by,
            acknowledged_at,
            created_at,
            dedup_key,
            occurrence_count,
            first_seen,
            last_seen,
            priority,
        })
    }
}
