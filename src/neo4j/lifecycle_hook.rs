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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{
        LifecycleActionType, LifecycleHook, LifecycleScope, UpdateLifecycleHookRequest,
    };
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;

    fn make_hook(name: &str, scope: LifecycleScope, on_status: &str) -> LifecycleHook {
        LifecycleHook::new(
            name.to_string(),
            scope,
            on_status.to_string(),
            LifecycleActionType::EmitAlert,
            serde_json::json!({"level": "info"}),
        )
    }

    // ── snake_case_from_debug tests ──

    #[test]
    fn test_snake_case_from_debug_cascade_children() {
        assert_eq!(snake_case_from_debug("CascadeChildren"), "cascade_children");
    }

    #[test]
    fn test_snake_case_from_debug_mcp_call() {
        assert_eq!(snake_case_from_debug("McpCall"), "mcp_call");
    }

    #[test]
    fn test_snake_case_from_debug_single_word() {
        assert_eq!(snake_case_from_debug("Task"), "task");
    }

    #[test]
    fn test_snake_case_from_debug_already_lowercase() {
        assert_eq!(snake_case_from_debug("task"), "task");
    }

    // ── CRUD operations via MockGraphStore ──

    #[tokio::test]
    async fn test_create_and_get_lifecycle_hook() {
        let store = MockGraphStore::new();
        let hook = make_hook("alert-on-complete", LifecycleScope::Task, "completed");
        let hook_id = hook.id;

        store.create_lifecycle_hook(&hook).await.unwrap();

        let retrieved = store.get_lifecycle_hook(hook_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "alert-on-complete");
        assert_eq!(retrieved.scope, LifecycleScope::Task);
        assert_eq!(retrieved.on_status, "completed");
    }

    #[tokio::test]
    async fn test_get_nonexistent_hook_returns_none() {
        let store = MockGraphStore::new();
        let result = store.get_lifecycle_hook(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_list_lifecycle_hooks_no_filter() {
        let store = MockGraphStore::new();
        let h1 = make_hook("hook-1", LifecycleScope::Task, "completed");
        let h2 = make_hook("hook-2", LifecycleScope::Plan, "in_progress");
        store.create_lifecycle_hook(&h1).await.unwrap();
        store.create_lifecycle_hook(&h2).await.unwrap();

        let hooks = store.list_lifecycle_hooks(None).await.unwrap();
        assert_eq!(hooks.len(), 2);
    }

    #[tokio::test]
    async fn test_list_lifecycle_hooks_with_project_filter() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        let mut h1 = make_hook("project-hook", LifecycleScope::Task, "completed");
        h1.project_id = Some(project_id);

        let h2 = make_hook("global-hook", LifecycleScope::Task, "completed");

        let mut h3 = make_hook("other-project-hook", LifecycleScope::Task, "completed");
        h3.project_id = Some(Uuid::new_v4());

        store.create_lifecycle_hook(&h1).await.unwrap();
        store.create_lifecycle_hook(&h2).await.unwrap();
        store.create_lifecycle_hook(&h3).await.unwrap();

        let hooks = store.list_lifecycle_hooks(Some(project_id)).await.unwrap();
        // Should include the project-specific hook and the global hook (no project_id)
        assert_eq!(hooks.len(), 2);
    }

    #[tokio::test]
    async fn test_update_lifecycle_hook() {
        let store = MockGraphStore::new();
        let hook = make_hook("original", LifecycleScope::Task, "completed");
        let hook_id = hook.id;
        store.create_lifecycle_hook(&hook).await.unwrap();

        let updates = UpdateLifecycleHookRequest {
            name: Some("updated-name".to_string()),
            description: Some(Some("new description".to_string())),
            on_status: Some("in_progress".to_string()),
            action_config: None,
            priority: Some(50),
            enabled: Some(false),
        };
        store
            .update_lifecycle_hook(hook_id, &updates)
            .await
            .unwrap();

        let updated = store.get_lifecycle_hook(hook_id).await.unwrap().unwrap();
        assert_eq!(updated.name, "updated-name");
        assert_eq!(updated.description.as_deref(), Some("new description"));
        assert_eq!(updated.on_status, "in_progress");
        assert_eq!(updated.priority, 50);
        assert!(!updated.enabled);
    }

    #[tokio::test]
    async fn test_delete_lifecycle_hook() {
        let store = MockGraphStore::new();
        let hook = make_hook("deletable", LifecycleScope::Task, "completed");
        let hook_id = hook.id;
        store.create_lifecycle_hook(&hook).await.unwrap();

        store.delete_lifecycle_hook(hook_id).await.unwrap();

        let result = store.get_lifecycle_hook(hook_id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete_builtin_hook_returns_error() {
        let store = MockGraphStore::new();
        let mut hook = make_hook("builtin-hook", LifecycleScope::Task, "completed");
        hook.builtin = true;
        let hook_id = hook.id;
        store.create_lifecycle_hook(&hook).await.unwrap();

        let result = store.delete_lifecycle_hook(hook_id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("builtin"));

        // Verify it still exists
        let still_exists = store.get_lifecycle_hook(hook_id).await.unwrap();
        assert!(still_exists.is_some());
    }

    #[tokio::test]
    async fn test_list_hooks_for_scope_filters_correctly() {
        let store = MockGraphStore::new();

        let h1 = make_hook("task-completed", LifecycleScope::Task, "completed");
        let h2 = make_hook("task-in-progress", LifecycleScope::Task, "in_progress");
        let h3 = make_hook("plan-completed", LifecycleScope::Plan, "completed");
        let mut h4 = make_hook("disabled-hook", LifecycleScope::Task, "completed");
        h4.enabled = false;

        store.create_lifecycle_hook(&h1).await.unwrap();
        store.create_lifecycle_hook(&h2).await.unwrap();
        store.create_lifecycle_hook(&h3).await.unwrap();
        store.create_lifecycle_hook(&h4).await.unwrap();

        // Only enabled Task hooks with on_status "completed"
        let result = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "task-completed");
    }

    #[tokio::test]
    async fn test_list_hooks_for_scope_includes_global_and_project_hooks() {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();

        // Global hook (no project_id)
        let h1 = make_hook("global", LifecycleScope::Task, "completed");

        // Project-specific hook
        let mut h2 = make_hook("project-specific", LifecycleScope::Task, "completed");
        h2.project_id = Some(project_id);

        // Hook for a different project
        let mut h3 = make_hook("other-project", LifecycleScope::Task, "completed");
        h3.project_id = Some(Uuid::new_v4());

        store.create_lifecycle_hook(&h1).await.unwrap();
        store.create_lifecycle_hook(&h2).await.unwrap();
        store.create_lifecycle_hook(&h3).await.unwrap();

        // With project_id: should get global + project-specific
        let result = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", Some(project_id))
            .await
            .unwrap();
        assert_eq!(result.len(), 2);

        // Without project_id: should get only global
        let result = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "global");
    }

    #[tokio::test]
    async fn test_list_hooks_for_scope_sorted_by_priority() {
        let store = MockGraphStore::new();

        let mut h1 = make_hook("low-priority", LifecycleScope::Task, "completed");
        h1.priority = 200;
        let mut h2 = make_hook("high-priority", LifecycleScope::Task, "completed");
        h2.priority = 10;
        let mut h3 = make_hook("mid-priority", LifecycleScope::Task, "completed");
        h3.priority = 100;

        store.create_lifecycle_hook(&h1).await.unwrap();
        store.create_lifecycle_hook(&h2).await.unwrap();
        store.create_lifecycle_hook(&h3).await.unwrap();

        let result = store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", None)
            .await
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "high-priority");
        assert_eq!(result[1].name, "mid-priority");
        assert_eq!(result[2].name, "low-priority");
    }
}
