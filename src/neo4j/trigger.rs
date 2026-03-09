//! Neo4j Trigger operations — trigger persistence and firing history

use super::client::Neo4jClient;
use crate::runner::{Trigger, TriggerFiring, TriggerType};
use anyhow::Result;
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

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
                created_at: datetime($created_at)
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
        .param("created_at", trigger.created_at.to_rfc3339());

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
    pub async fn list_all_triggers_impl(
        &self,
        trigger_type: Option<&str>,
    ) -> Result<Vec<Trigger>> {
        let cypher = if let Some(tt) = trigger_type {
            format!(
                "MATCH (t:Trigger) WHERE t.trigger_type = '{}' RETURN t ORDER BY t.created_at DESC",
                tt
            )
        } else {
            "MATCH (t:Trigger) RETURN t ORDER BY t.created_at DESC".to_string()
        };

        let q = query(&cypher);
        let mut result = self.graph.execute(q).await?;
        let mut triggers = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            triggers.push(self.node_to_trigger(&node)?);
        }
        Ok(triggers)
    }

    /// Update a trigger's enabled status, config, or cooldown.
    pub async fn update_trigger_impl(
        &self,
        trigger_id: Uuid,
        enabled: Option<bool>,
        config: Option<serde_json::Value>,
        cooldown_secs: Option<u64>,
    ) -> Result<Option<Trigger>> {
        let mut set_clauses = Vec::new();
        if enabled.is_some() {
            set_clauses.push("t.enabled = $enabled".to_string());
        }
        if config.is_some() {
            set_clauses.push("t.config = $config".to_string());
        }
        if cooldown_secs.is_some() {
            set_clauses.push("t.cooldown_secs = $cooldown_secs".to_string());
        }

        if set_clauses.is_empty() {
            return self.get_trigger_impl(trigger_id).await;
        }

        let cypher = format!(
            "MATCH (t:Trigger {{id: $id}}) SET {} RETURN t",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher).param("id", trigger_id.to_string());

        if let Some(e) = enabled {
            q = q.param("enabled", e);
        }
        if let Some(c) = config {
            q = q.param("config", serde_json::to_string(&c).unwrap_or_default());
        }
        if let Some(cd) = cooldown_secs {
            q = q.param("cooldown_secs", cd as i64);
        }

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
                source_payload: $source_payload
            })
            CREATE (f)-[:FIRED_BY]->(t)
            SET t.fire_count = t.fire_count + 1,
                t.last_fired = datetime($fired_at)
            "#,
        );

        if let Some(run_id) = firing.plan_run_id {
            cypher.push_str(&format!(
                r#"
                WITH f
                MATCH (r:PlanRun {{run_id: '{}'}})
                CREATE (f)-[:STARTED]->(r)
                "#,
                run_id
            ));
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
            );

        self.graph.run(q).await?;
        Ok(())
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
        })
    }

    /// Convert a Neo4j node to a TriggerFiring.
    fn node_to_trigger_firing(&self, node: &neo4rs::Node) -> Result<TriggerFiring> {
        let id: String = node.get("id")?;
        let trigger_id: String = node.get("trigger_id")?;
        let fired_at: String = node.get("fired_at")?;
        let source_payload: Option<String> = node.get("source_payload").ok();
        let plan_run_id: Option<String> = node.get("plan_run_id").ok();

        Ok(TriggerFiring {
            id: id.parse()?,
            trigger_id: trigger_id.parse()?,
            plan_run_id: plan_run_id.and_then(|s| s.parse().ok()),
            fired_at: fired_at.parse()?,
            source_payload: source_payload
                .and_then(|s| if s.is_empty() { None } else { serde_json::from_str(&s).ok() }),
        })
    }
}
