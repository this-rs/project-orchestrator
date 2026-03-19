//! Neo4j EventTrigger operations — persistent event-to-protocol triggers
//!
//! CRUD for [`EventTrigger`] nodes and their `:TRIGGERS` relationship to Protocol nodes.

use super::client::Neo4jClient;
use crate::events::trigger::EventTrigger;
use anyhow::Result;
use chrono::Utc;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Conversion helpers
    // ========================================================================

    /// Convert a Neo4j node to an [`EventTrigger`].
    fn node_to_event_trigger(&self, node: &neo4rs::Node) -> Result<EventTrigger> {
        let id: String = node.get("id")?;
        let name: String = node.get("name")?;
        let protocol_id: String = node.get("protocol_id")?;
        let entity_type_pattern: Option<String> = node
            .get::<String>("entity_type_pattern")
            .ok()
            .filter(|s| !s.is_empty());
        let action_pattern: Option<String> = node
            .get::<String>("action_pattern")
            .ok()
            .filter(|s| !s.is_empty());
        let payload_conditions_str: Option<String> = node
            .get::<String>("payload_conditions")
            .ok()
            .filter(|s| !s.is_empty());
        let cooldown_secs: i64 = node.get("cooldown_secs").unwrap_or(0);
        let enabled: bool = node.get("enabled").unwrap_or(true);
        let project_scope: Option<String> = node
            .get::<String>("project_scope")
            .ok()
            .filter(|s| !s.is_empty());
        let created_at: String = node.get("created_at")?;
        let updated_at: String = node.get("updated_at")?;

        Ok(EventTrigger {
            id: id.parse()?,
            name,
            protocol_id: protocol_id.parse()?,
            entity_type_pattern,
            action_pattern,
            payload_conditions: payload_conditions_str
                .and_then(|s| serde_json::from_str(&s).ok()),
            cooldown_secs: cooldown_secs as u32,
            enabled,
            project_scope: project_scope.and_then(|s| s.parse().ok()),
            created_at: created_at.parse()?,
            updated_at: updated_at.parse()?,
        })
    }

    // ========================================================================
    // CRUD operations
    // ========================================================================

    /// Create an EventTrigger node in Neo4j.
    ///
    /// Returns the generated UUID.
    pub async fn create_event_trigger(&self, trigger: &EventTrigger) -> Result<Uuid> {
        let now = Utc::now();
        let q = query(
            r#"
            CREATE (t:EventTrigger {
                id: $id,
                name: $name,
                protocol_id: $protocol_id,
                entity_type_pattern: $entity_type_pattern,
                action_pattern: $action_pattern,
                payload_conditions: $payload_conditions,
                cooldown_secs: $cooldown_secs,
                enabled: $enabled,
                project_scope: $project_scope,
                created_at: $created_at,
                updated_at: $updated_at
            })
            "#,
        )
        .param("id", trigger.id.to_string())
        .param("name", trigger.name.clone())
        .param("protocol_id", trigger.protocol_id.to_string())
        .param(
            "entity_type_pattern",
            trigger.entity_type_pattern.clone().unwrap_or_default(),
        )
        .param(
            "action_pattern",
            trigger.action_pattern.clone().unwrap_or_default(),
        )
        .param(
            "payload_conditions",
            trigger
                .payload_conditions
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_default(),
        )
        .param("cooldown_secs", trigger.cooldown_secs as i64)
        .param("enabled", trigger.enabled)
        .param(
            "project_scope",
            trigger
                .project_scope
                .map(|u| u.to_string())
                .unwrap_or_default(),
        )
        .param("created_at", now.to_rfc3339())
        .param("updated_at", now.to_rfc3339());

        self.graph.run(q).await?;
        Ok(trigger.id)
    }

    /// Get an EventTrigger by its UUID.
    pub async fn get_event_trigger(&self, id: Uuid) -> Result<Option<EventTrigger>> {
        let q = query(
            r#"
            MATCH (t:EventTrigger {id: $id})
            RETURN t
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            Ok(Some(self.node_to_event_trigger(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List EventTriggers, optionally filtered by project scope and/or enabled status.
    pub async fn list_event_triggers(
        &self,
        project_scope: Option<Uuid>,
        enabled_only: bool,
    ) -> Result<Vec<EventTrigger>> {
        let mut where_clauses = Vec::new();
        if project_scope.is_some() {
            // Match triggers scoped to this project OR global triggers (empty project_scope)
            where_clauses.push(
                "(t.project_scope = $project_scope OR t.project_scope = '')".to_string(),
            );
        }
        if enabled_only {
            where_clauses.push("t.enabled = true".to_string());
        }

        let where_str = if where_clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_clauses.join(" AND "))
        };

        let cypher = format!(
            "MATCH (t:EventTrigger){} RETURN t ORDER BY t.created_at DESC",
            where_str
        );

        let mut q = query(&cypher);
        if let Some(scope) = project_scope {
            q = q.param("project_scope", scope.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut triggers = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("t")?;
            match self.node_to_event_trigger(&node) {
                Ok(t) => triggers.push(t),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to parse EventTrigger node");
                }
            }
        }
        Ok(triggers)
    }

    /// Update an EventTrigger's fields. Only provided fields are updated.
    ///
    /// Returns `true` if the trigger was found and updated, `false` if not found.
    pub async fn update_event_trigger(
        &self,
        id: Uuid,
        enabled: Option<bool>,
        name: Option<String>,
        entity_type_pattern: Option<Option<String>>,
        action_pattern: Option<Option<String>>,
        payload_conditions: Option<Option<serde_json::Value>>,
        cooldown_secs: Option<u32>,
        project_scope: Option<Option<Uuid>>,
    ) -> Result<bool> {
        let mut set_clauses = vec!["t.updated_at = $updated_at".to_string()];

        if enabled.is_some() {
            set_clauses.push("t.enabled = $enabled".to_string());
        }
        if name.is_some() {
            set_clauses.push("t.name = $name".to_string());
        }
        if entity_type_pattern.is_some() {
            set_clauses.push("t.entity_type_pattern = $entity_type_pattern".to_string());
        }
        if action_pattern.is_some() {
            set_clauses.push("t.action_pattern = $action_pattern".to_string());
        }
        if payload_conditions.is_some() {
            set_clauses.push("t.payload_conditions = $payload_conditions".to_string());
        }
        if cooldown_secs.is_some() {
            set_clauses.push("t.cooldown_secs = $cooldown_secs".to_string());
        }
        if project_scope.is_some() {
            set_clauses.push("t.project_scope = $project_scope".to_string());
        }

        let cypher = format!(
            "MATCH (t:EventTrigger {{id: $id}}) SET {} RETURN t",
            set_clauses.join(", ")
        );

        let mut q = query(&cypher)
            .param("id", id.to_string())
            .param("updated_at", Utc::now().to_rfc3339());

        if let Some(e) = enabled {
            q = q.param("enabled", e);
        }
        if let Some(n) = name {
            q = q.param("name", n);
        }
        if let Some(etp) = entity_type_pattern {
            q = q.param("entity_type_pattern", etp.unwrap_or_default());
        }
        if let Some(ap) = action_pattern {
            q = q.param("action_pattern", ap.unwrap_or_default());
        }
        if let Some(pc) = payload_conditions {
            q = q.param(
                "payload_conditions",
                pc.map(|v| serde_json::to_string(&v).unwrap_or_default())
                    .unwrap_or_default(),
            );
        }
        if let Some(cd) = cooldown_secs {
            q = q.param("cooldown_secs", cd as i64);
        }
        if let Some(ps) = project_scope {
            q = q.param(
                "project_scope",
                ps.map(|u| u.to_string()).unwrap_or_default(),
            );
        }

        let mut result = self.graph.execute(q).await?;
        Ok(result.next().await?.is_some())
    }

    /// Delete an EventTrigger by its UUID.
    ///
    /// Returns `true` if a node was deleted, `false` if not found.
    pub async fn delete_event_trigger(&self, id: Uuid) -> Result<bool> {
        // DETACH DELETE + RETURN doesn't work well — use a two-step approach:
        // check existence first, then delete.
        let check = query(
            r#"
            MATCH (t:EventTrigger {id: $id})
            RETURN t.id AS id
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(check).await?;
        let exists = result.next().await?.is_some();

        if exists {
            let del = query(
                r#"
                MATCH (t:EventTrigger {id: $id})
                DETACH DELETE t
                "#,
            )
            .param("id", id.to_string());
            self.graph.run(del).await?;
        }

        Ok(exists)
    }

    /// Create a `:TRIGGERS` relationship from an EventTrigger to a Protocol.
    ///
    /// This relationship indicates that when the trigger fires, the linked
    /// protocol should be activated.
    pub async fn link_trigger_to_protocol(
        &self,
        trigger_id: Uuid,
        protocol_id: Uuid,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (t:EventTrigger {id: $trigger_id})
            MATCH (p:Protocol {id: $protocol_id})
            MERGE (t)-[:TRIGGERS]->(p)
            "#,
        )
        .param("trigger_id", trigger_id.to_string())
        .param("protocol_id", protocol_id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }
}
