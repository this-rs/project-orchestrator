//! Neo4j Trigger operations — trigger persistence and firing history

use super::client::Neo4jClient;
use crate::runner::{
    PlanContent, SignalReservation, Trigger, TriggerAuthor, TriggerFiring, TriggerType,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

/// How long a reserved trigger signal is remembered: the same key seen again
/// within this window is a duplicate.
pub const TRIGGER_SIGNAL_RETENTION_HOURS: i64 = 24;

/// A trigger's author as stored on the node: JSON, `""` for none.
fn author_param(author: Option<&TriggerAuthor>) -> String {
    author
        .and_then(|a| serde_json::to_string(a).ok())
        .unwrap_or_default()
}

impl Neo4jClient {
    /// Create a Trigger node and link it to a Plan via (:Trigger)-[:TRIGGERS]->(:Plan).
    pub async fn create_trigger_impl(&self, trigger: &Trigger) -> Result<Trigger> {
        let q = query(
            r#"
            MATCH (p:Plan {id: $plan_id})
            CREATE (t:Trigger {
                id: $id,
                plan_id: $plan_id,
                trigger_type: $trigger_type,
                config: $config,
                enabled: $enabled,
                cooldown_secs: $cooldown_secs,
                fire_count: 0,
                created_at: datetime($created_at),
                author: $author
            })
            CREATE (t)-[:TRIGGERS]->(p)
            RETURN t
            "#,
        )
        .param("id", trigger.id.to_string())
        .param("plan_id", trigger.plan_id.to_string())
        .param("trigger_type", trigger.trigger_type.to_string())
        .param(
            "config",
            serde_json::to_string(&trigger.config).unwrap_or_default(),
        )
        .param("enabled", trigger.enabled)
        .param("cooldown_secs", trigger.cooldown_secs as i64)
        .param("created_at", trigger.created_at.to_rfc3339())
        .param("author", author_param(trigger.author.as_ref()));

        self.graph.run(q).await?;
        Ok(trigger.clone())
    }

    /// Get a Trigger by its UUID.
    pub async fn get_trigger_impl(&self, trigger_id: Uuid) -> Result<Option<Trigger>> {
        let q = query(
            r#"
            MATCH (t:Trigger {id: $id})
            RETURN t
            "#,
        )
        .param("id", trigger_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_trigger(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List all triggers for a given plan.
    pub async fn list_triggers_impl(&self, plan_id: Uuid) -> Result<Vec<Trigger>> {
        let q = query(
            r#"
            MATCH (t:Trigger {plan_id: $plan_id})
            RETURN t
            ORDER BY t.created_at DESC
            "#,
        )
        .param("plan_id", plan_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut triggers = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            triggers.push(self.node_to_trigger(&node)?);
        }
        Ok(triggers)
    }

    /// List all triggers, optionally filtered by type.
    pub async fn list_all_triggers_impl(&self, trigger_type: Option<&str>) -> Result<Vec<Trigger>> {
        let cypher = if trigger_type.is_some() {
            "MATCH (t:Trigger) WHERE t.trigger_type = $trigger_type RETURN t ORDER BY t.created_at DESC"
        } else {
            "MATCH (t:Trigger) RETURN t ORDER BY t.created_at DESC"
        };

        let mut q = query(cypher);
        if let Some(tt) = trigger_type {
            q = q.param("trigger_type", tt.to_string());
        }
        let mut result = self.graph.execute(q).await?;
        let mut triggers = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            triggers.push(self.node_to_trigger(&node)?);
        }
        Ok(triggers)
    }

    /// Disable a trigger; `reason` when the system does it (`""` stored for
    /// none, read back as `None`).
    pub async fn disable_trigger_impl(
        &self,
        trigger_id: Uuid,
        reason: Option<&str>,
    ) -> Result<Option<Trigger>> {
        let q = query(
            r#"
            MATCH (t:Trigger {id: $id})
            SET t.enabled = false, t.disabled_reason = $reason
            RETURN t
            "#,
        )
        .param("id", trigger_id.to_string())
        .param("reason", reason.unwrap_or_default().to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_trigger(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Delete a trigger and its firing history.
    pub async fn delete_trigger_impl(&self, trigger_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:Trigger {id: $id})
            OPTIONAL MATCH (f:TriggerFiring)-[:FIRED_BY]->(t)
            DETACH DELETE f, t
            "#,
        )
        .param("id", trigger_id.to_string());

        self.graph.run(q).await?;
        let signals = query("MATCH (s:TriggerSignal {trigger_id: $id}) DELETE s")
            .param("id", trigger_id.to_string());
        self.graph.run(signals).await?;
        Ok(())
    }

    /// Record a trigger firing event.
    pub async fn record_trigger_firing_impl(&self, firing: &TriggerFiring) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (t:Trigger {id: $trigger_id})
            CREATE (f:TriggerFiring {
                id: $id,
                trigger_id: $trigger_id,
                fired_at: datetime($fired_at),
                source_payload: $source_payload,
                plan_run_id: $plan_run_id,
                start_error: $start_error
            })
            CREATE (f)-[:FIRED_BY]->(t)
            SET t.fire_count = t.fire_count + 1
            "#,
        );

        if firing.plan_run_id.is_some() {
            cypher.push_str(
                r#"
                WITH f
                MATCH (r:PlanRun {run_id: $plan_run_id})
                CREATE (f)-[:STARTED]->(r)
                "#,
            );
        }

        let q = query(&cypher)
            .param("id", firing.id.to_string())
            .param("trigger_id", firing.trigger_id.to_string())
            .param("fired_at", firing.fired_at.to_rfc3339())
            .param(
                "source_payload",
                firing
                    .source_payload
                    .as_ref()
                    .map(|p| serde_json::to_string(p).unwrap_or_default())
                    .unwrap_or_default(),
            )
            .param(
                "plan_run_id",
                firing
                    .plan_run_id
                    .map(|id| id.to_string())
                    .unwrap_or_default(),
            )
            .param(
                "start_error",
                firing.start_error.clone().unwrap_or_default(),
            );

        self.graph.run(q).await?;
        Ok(())
    }

    /// Enable the trigger and record its author, in one write.
    pub async fn enable_trigger_as_impl(
        &self,
        trigger_id: Uuid,
        author: &TriggerAuthor,
    ) -> Result<Option<Trigger>> {
        let q = query(
            r#"
            MATCH (t:Trigger {id: $id})
            SET t.enabled = true, t.author = $author
            REMOVE t.disabled_reason
            RETURN t
            "#,
        )
        .param("id", trigger_id.to_string())
        .param("author", author_param(Some(author)));

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_trigger(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Reserve the signal `key` of a trigger: one `TriggerSignal` per
    /// `(trigger_id, key)` (unique constraint `trigger_signal_key`), and the
    /// cooldown read and `last_fired` written in the same statement.
    ///
    /// The first `SET` takes the write lock on the Trigger node before anything
    /// is read, so two reservations of one trigger are serialized: the second
    /// sees the signal and the `last_fired` the first committed. A signal
    /// older than 24 h no longer counts (a delivery replayed after that starts
    /// a run again); such signals are purged after each reservation.
    pub async fn reserve_trigger_signal_impl(
        &self,
        trigger_id: Uuid,
        key: &str,
        cooldown_secs: u64,
    ) -> Result<SignalReservation> {
        let q = query(
            r#"
            MATCH (t:Trigger {id: $trigger_id})
            SET t.reservation_lock = true
            WITH t
            OPTIONAL MATCH (seen:TriggerSignal {trigger_id: $trigger_id, key: $key})
            WHERE seen.reserved_at >= datetime() - duration({hours: $retention_hours})
            WITH t, seen IS NULL AS fresh,
                 ($cooldown_secs = 0 OR t.last_fired IS NULL
                  OR t.last_fired + duration({seconds: $cooldown_secs}) <= datetime()) AS cooled
            FOREACH (_ IN CASE WHEN fresh AND cooled THEN [1] ELSE [] END |
                MERGE (s:TriggerSignal {trigger_id: $trigger_id, key: $key})
                SET s.reserved_at = datetime(), t.last_fired = datetime())
            REMOVE t.reservation_lock
            RETURN fresh, cooled
            "#,
        )
        .param("trigger_id", trigger_id.to_string())
        .param("key", key.to_string())
        .param("cooldown_secs", cooldown_secs as i64)
        .param("retention_hours", TRIGGER_SIGNAL_RETENTION_HOURS);

        let mut result = self.graph.execute(q).await?;
        let reservation = match result.next().await? {
            Some(row) => match (row.get::<bool>("fresh")?, row.get::<bool>("cooled")?) {
                (false, _) => SignalReservation::Duplicate,
                (true, false) => SignalReservation::Cooldown,
                (true, true) => SignalReservation::Reserved,
            },
            None => SignalReservation::Duplicate,
        };
        drop(result);

        let purge = query(
            r#"
            MATCH (s:TriggerSignal)
            WHERE s.reserved_at < datetime() - duration({hours: $retention_hours})
            WITH s LIMIT 1000
            DELETE s
            "#,
        )
        .param("retention_hours", TRIGGER_SIGNAL_RETENTION_HOURS);
        if let Err(e) = self.graph.run(purge).await {
            tracing::warn!("Purge of old trigger signals failed (retried next time): {e:#}");
        }
        Ok(reservation)
    }

    /// List trigger firings for a given trigger, ordered by fired_at desc.
    pub async fn list_trigger_firings_impl(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> Result<Vec<TriggerFiring>> {
        let q = query(
            r#"
            MATCH (f:TriggerFiring {trigger_id: $trigger_id})
            RETURN f
            ORDER BY f.fired_at DESC
            LIMIT $limit
            "#,
        )
        .param("trigger_id", trigger_id.to_string())
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut firings = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("f")?;
            firings.push(self.node_to_trigger_firing(&node)?);
        }
        Ok(firings)
    }

    /// Convert a Neo4j node to a Trigger.
    fn node_to_trigger(&self, node: &neo4rs::Node) -> Result<Trigger> {
        let id: String = node.get("id")?;
        let plan_id: String = node.get("plan_id")?;
        let trigger_type: String = node.get("trigger_type")?;
        let config_str: String = node.get("config").unwrap_or_default();
        let enabled: bool = node.get("enabled").unwrap_or(true);
        let cooldown_secs: i64 = node.get("cooldown_secs").unwrap_or(0);
        let fire_count: i64 = node.get("fire_count").unwrap_or(0);
        let created_at: String = node.get("created_at")?;
        let last_fired: Option<String> = node.get("last_fired").ok();
        let author: Option<String> = node.get("author").ok();
        let disabled_reason: Option<String> = node.get("disabled_reason").ok();

        let tt = match trigger_type.as_str() {
            "schedule" => TriggerType::Schedule,
            "webhook" => TriggerType::Webhook,
            "event" => TriggerType::Event,
            "chat" => TriggerType::Chat,
            _ => TriggerType::Event,
        };

        Ok(Trigger {
            id: id.parse()?,
            plan_id: plan_id.parse()?,
            trigger_type: tt,
            config: serde_json::from_str(&config_str).unwrap_or(serde_json::Value::Null),
            enabled,
            cooldown_secs: cooldown_secs as u64,
            last_fired: last_fired.and_then(|s| s.parse::<DateTime<Utc>>().ok()),
            fire_count: fire_count as u64,
            created_at: created_at.parse()?,
            author: author
                .filter(|a| !a.is_empty())
                .and_then(|a| serde_json::from_str(&a).ok()),
            // A disabled_reason only means something on a disabled trigger.
            disabled_reason: disabled_reason.filter(|r| !r.is_empty() && !enabled),
        })
    }

    /// Mark the plan `content` belongs to as written by a third-party session
    /// (`third_party_written_at = datetime()`), in one statement.
    pub async fn mark_third_party_write_impl(&self, content: PlanContent) -> Result<Option<Uuid>> {
        let (pattern, id) = match content {
            PlanContent::Plan(id) => ("MATCH (p:Plan {id: $id})", id),
            PlanContent::Task(id) => ("MATCH (p:Plan)-[:HAS_TASK]->(:Task {id: $id})", id),
            PlanContent::Step(id) => (
                "MATCH (p:Plan)-[:HAS_TASK]->(:Task)-[:HAS_STEP]->(:Step {id: $id})",
                id,
            ),
            PlanContent::Constraint(id) => (
                "MATCH (p:Plan)-[:CONSTRAINED_BY]->(:Constraint {id: $id})",
                id,
            ),
            PlanContent::Decision(id) => (
                "MATCH (p:Plan)-[:HAS_TASK]->(:Task)-[:INFORMED_BY]->(:Decision {id: $id})",
                id,
            ),
        };
        // `pattern` is one of the fixed strings above; the id is a parameter.
        let cypher = format!(
            "{pattern} WITH DISTINCT p SET p.third_party_written_at = datetime() RETURN p.id AS plan_id"
        );
        let mut result = self
            .graph
            .execute(query(&cypher).param("id", id.to_string()))
            .await?;
        match result.next().await? {
            Some(row) => Ok(row.get::<String>("plan_id")?.parse().ok()),
            None => Ok(None),
        }
    }

    /// When a third-party session last wrote the plan (see
    /// [`Self::mark_third_party_write_impl`]).
    pub async fn plan_third_party_written_at_impl(
        &self,
        plan_id: Uuid,
    ) -> Result<Option<DateTime<Utc>>> {
        let q = query("MATCH (p:Plan {id: $id}) RETURN toString(p.third_party_written_at) AS at")
            .param("id", plan_id.to_string());
        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => Ok(row
                .get::<Option<String>>("at")
                .ok()
                .flatten()
                .and_then(|s| s.parse::<DateTime<Utc>>().ok())),
            None => Ok(None),
        }
    }

    /// Convert a Neo4j node to a TriggerFiring.
    fn node_to_trigger_firing(&self, node: &neo4rs::Node) -> Result<TriggerFiring> {
        let id: String = node.get("id")?;
        let trigger_id: String = node.get("trigger_id")?;
        let fired_at: String = node.get("fired_at")?;
        let source_payload: Option<String> = node.get("source_payload").ok();
        let plan_run_id: Option<String> = node.get("plan_run_id").ok();
        let start_error: Option<String> = node.get("start_error").ok();

        Ok(TriggerFiring {
            id: id.parse()?,
            trigger_id: trigger_id.parse()?,
            plan_run_id: plan_run_id.and_then(|s| s.parse().ok()),
            fired_at: fired_at.parse()?,
            source_payload: source_payload.and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    serde_json::from_str(&s).ok()
                }
            }),
            start_error: start_error.filter(|e| !e.is_empty()),
        })
    }
}
