//! Neo4j LifecycleHook operations

use super::client::Neo4jClient;
use crate::lifecycle::{
    LifecycleActionType, LifecycleHook, LifecycleScope, UpdateLifecycleHookRequest,
};
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // LifecycleHook operations
    // ========================================================================

    /// Create a lifecycle hook
    pub async fn create_lifecycle_hook(&self, hook: &LifecycleHook) -> Result<()> {
        let now = hook.created_at.to_rfc3339();
        let action_config_str = serde_json::to_string(&hook.action_config)?;
        let project_id_str = hook.project_id.map(|id| id.to_string());

        let q = query(
            r#"
            CREATE (h:LifecycleHook {
                id: $id,
                name: $name,
                description: $description,
                scope: $scope,
                on_status: $on_status,
                action_type: $action_type,
                action_config: $action_config,
                priority: $priority,
                enabled: $enabled,
                builtin: $builtin,
                project_id: $project_id,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at)
            })
            "#,
        )
        .param("id", hook.id.to_string())
        .param("name", hook.name.clone())
        .param("description", hook.description.clone().unwrap_or_default())
        .param("scope", format!("{:?}", hook.scope))
        .param("on_status", hook.on_status.clone())
        .param("action_type", format!("{:?}", hook.action_type))
        .param("action_config", action_config_str)
        .param("priority", hook.priority as i64)
        .param("enabled", hook.enabled)
        .param("builtin", hook.builtin)
        .param("project_id", project_id_str.unwrap_or_default())
        .param("created_at", now.clone())
        .param("updated_at", now);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Get a lifecycle hook by ID
    pub async fn get_lifecycle_hook(&self, id: Uuid) -> Result<Option<LifecycleHook>> {
        let q = query(
            r#"
            MATCH (h:LifecycleHook {id: $id})
            RETURN h
            "#,
        )
        .param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("h")?;
            Ok(Some(parse_lifecycle_hook_node(&node)?))
        } else {
            Ok(None)
        }
    }

    /// List lifecycle hooks, optionally filtered by project_id.
    /// If project_id is None, returns all hooks.
    pub async fn list_lifecycle_hooks(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<Vec<LifecycleHook>> {
        let (cypher, q) = match project_id {
            Some(pid) => {
                let cypher = r#"
                    MATCH (h:LifecycleHook)
                    WHERE h.project_id = $project_id OR h.project_id = ''
                    RETURN h
                    ORDER BY h.priority ASC
                "#;
                let q = query(cypher).param("project_id", pid.to_string());
                (cypher, q)
            }
            None => {
                let cypher = r#"
                    MATCH (h:LifecycleHook)
                    RETURN h
                    ORDER BY h.priority ASC
                "#;
                let q = query(cypher);
                (cypher, q)
            }
        };
        let _ = cypher; // suppress unused warning

        let mut result = self.graph.execute(q).await?;
        let mut hooks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("h")?;
            hooks.push(parse_lifecycle_hook_node(&node)?);
        }

        Ok(hooks)
    }

    /// Update a lifecycle hook
    pub async fn update_lifecycle_hook(
        &self,
        id: Uuid,
        updates: &UpdateLifecycleHookRequest,
    ) -> Result<()> {
        let mut set_clauses = vec!["h.updated_at = datetime($now)"];

        if updates.name.is_some() {
            set_clauses.push("h.name = $name");
        }
        if updates.description.is_some() {
            set_clauses.push("h.description = $description");
        }
        if updates.on_status.is_some() {
            set_clauses.push("h.on_status = $on_status");
        }
        if updates.action_config.is_some() {
            set_clauses.push("h.action_config = $action_config");
        }
        if updates.priority.is_some() {
            set_clauses.push("h.priority = $priority");
        }
        if updates.enabled.is_some() {
            set_clauses.push("h.enabled = $enabled");
        }

        let cypher = format!(
            "MATCH (h:LifecycleHook {{id: $id}}) SET {}",
            set_clauses.join(", ")
        );
        let now = chrono::Utc::now().to_rfc3339();
        let mut q = query(&cypher).param("id", id.to_string()).param("now", now);

        if let Some(name) = &updates.name {
            q = q.param("name", name.clone());
        }
        if let Some(description) = &updates.description {
            q = q.param("description", description.clone().unwrap_or_default());
        }
        if let Some(on_status) = &updates.on_status {
            q = q.param("on_status", on_status.clone());
        }
        if let Some(action_config) = &updates.action_config {
            q = q.param("action_config", serde_json::to_string(action_config)?);
        }
        if let Some(priority) = &updates.priority {
            q = q.param("priority", *priority as i64);
        }
        if let Some(enabled) = &updates.enabled {
            q = q.param("enabled", *enabled);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Delete a lifecycle hook. Builtin hooks cannot be deleted.
    pub async fn delete_lifecycle_hook(&self, id: Uuid) -> Result<()> {
        // First check if the hook is builtin
        let hook = self.get_lifecycle_hook(id).await?;
        if let Some(h) = &hook {
            if h.builtin {
                anyhow::bail!("Cannot delete builtin lifecycle hook");
            }
        }

        let q = query(
            r#"
            MATCH (h:LifecycleHook {id: $id})
            DETACH DELETE h
            "#,
        )
        .param("id", id.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List hooks matching a scope and on_status, including both project-specific
    /// and global hooks (project_id is empty). Ordered by priority ASC.
    pub async fn list_hooks_for_scope(
        &self,
        scope: &LifecycleScope,
        on_status: &str,
        project_id: Option<Uuid>,
    ) -> Result<Vec<LifecycleHook>> {
        let cypher = match project_id {
            Some(_) => {
                r#"
                MATCH (h:LifecycleHook)
                WHERE h.scope = $scope
                  AND h.on_status = $on_status
                  AND h.enabled = true
                  AND (h.project_id = $project_id OR h.project_id = '')
                RETURN h
                ORDER BY h.priority ASC
            "#
            }
            None => {
                r#"
                MATCH (h:LifecycleHook)
                WHERE h.scope = $scope
                  AND h.on_status = $on_status
                  AND h.enabled = true
                  AND h.project_id = ''
                RETURN h
                ORDER BY h.priority ASC
            "#
            }
        };

        let mut q = query(cypher)
            .param("scope", format!("{:?}", scope))
            .param("on_status", on_status.to_string());

        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }

        let mut result = self.graph.execute(q).await?;
        let mut hooks = Vec::new();

        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("h")?;
            hooks.push(parse_lifecycle_hook_node(&node)?);
        }

        Ok(hooks)
    }
}

/// Parse a Neo4j Node into a LifecycleHook
fn parse_lifecycle_hook_node(node: &neo4rs::Node) -> Result<LifecycleHook> {
    let scope_str = node.get::<String>("scope")?;
    let scope: LifecycleScope =
        serde_json::from_str(&format!("\"{}\"", snake_case_from_debug(&scope_str)))
            .unwrap_or(LifecycleScope::Task);

    let action_type_str = node.get::<String>("action_type")?;
    let action_type: LifecycleActionType =
        serde_json::from_str(&format!("\"{}\"", snake_case_from_debug(&action_type_str)))
            .unwrap_or(LifecycleActionType::EmitAlert);

    let action_config_str = node.get::<String>("action_config").unwrap_or_default();
    let action_config: serde_json::Value = serde_json::from_str(&action_config_str)
        .unwrap_or(serde_json::Value::Object(Default::default()));

    let project_id_str = node
        .get::<String>("project_id")
        .ok()
        .filter(|s| !s.is_empty());
    let project_id = project_id_str.and_then(|s| s.parse().ok());

    Ok(LifecycleHook {
        id: node.get::<String>("id")?.parse()?,
        name: node.get("name")?,
        description: node
            .get::<String>("description")
            .ok()
            .filter(|s| !s.is_empty()),
        scope,
        on_status: node.get("on_status")?,
        action_type,
        action_config,
        priority: node.get::<i64>("priority")? as i32,
        enabled: node.get::<bool>("enabled").unwrap_or(true),
        builtin: node.get::<bool>("builtin").unwrap_or(false),
        project_id,
        created_at: node
            .get::<String>("created_at")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(chrono::Utc::now),
        updated_at: node
            .get::<String>("updated_at")
            .ok()
            .and_then(|s| s.parse().ok()),
    })
}

/// Convert a Debug-formatted enum variant to snake_case for serde deserialization.
/// e.g. "CascadeChildren" -> "cascade_children", "McpCall" -> "mcp_call"
fn snake_case_from_debug(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}
