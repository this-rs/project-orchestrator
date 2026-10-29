//! Neo4j DayPlan operations.
//!
//! A `DayPlan` is a per-user, per-day ordered list of tasks: a single
//! `(:DayPlan {user_id, date, updated_at})` node with ordered
//! `[:INCLUDES {order}]->(:Task)` edges. It is the server-side home of the
//! "today plan": shared across clients and readable by the chat agent.
use crate::neo4j::traits::*;

use super::client::Neo4jClient;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Day plan operations
    // ========================================================================

    /// Read one user's ordered day plan as `(task_id, order)` pairs.
    ///
    /// Returns an empty list when the user has no plan for that date.
    pub async fn get_day_plan(&self, user_id: Uuid, date: &str) -> Result<Vec<(Uuid, u32)>> {
        let q = query(
            r#"
            MATCH (dp:DayPlan {user_id: $user_id, date: $date})
            OPTIONAL MATCH (dp)-[inc:INCLUDES]->(t:Task)
            RETURN t.id AS task_id, inc.order AS order
            ORDER BY inc.order
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("date", date.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut entries = Vec::new();
        while let Some(row) = result.next().await? {
            let task_id = row.get::<Option<String>>("task_id").ok().flatten();
            let order = row.get::<Option<i64>>("order").ok().flatten();
            if let (Some(id), Some(order)) = (task_id, order) {
                entries.push((id.parse()?, order as u32));
            }
        }
        Ok(entries)
    }

    /// Replace the whole day plan (idempotent). The position of each id in
    /// `ids` is its `order` (0-based). Unknown task ids are skipped; edges are
    /// rebuilt from scratch so the result is exactly the given list.
    pub async fn set_day_plan(&self, user_id: Uuid, date: &str, ids: &[Uuid]) -> Result<()> {
        let items: Vec<std::collections::HashMap<String, neo4rs::BoltType>> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let mut m = std::collections::HashMap::new();
                m.insert("id".to_string(), id.to_string().into());
                m.insert("order".to_string(), (i as i64).into());
                m
            })
            .collect();

        let q = query(
            r#"
            MERGE (dp:DayPlan {user_id: $user_id, date: $date})
            ON CREATE SET dp.updated_at = datetime($updated_at)
            ON MATCH SET dp.updated_at = datetime($updated_at)
            WITH dp
            OPTIONAL MATCH (dp)-[inc:INCLUDES]->(:Task)
            DELETE inc
            WITH dp
            UNWIND $items AS item
            MATCH (t:Task {id: item.id})
            CREATE (dp)-[:INCLUDES {order: item.order}]->(t)
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("date", date.to_string())
        .param("updated_at", chrono::Utc::now().to_rfc3339())
        .param("items", items);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Append one task to the end of the day plan. No-op when the task is
    /// already on the plan (or when the task id does not exist).
    pub async fn add_to_day(&self, user_id: Uuid, date: &str, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MERGE (dp:DayPlan {user_id: $user_id, date: $date})
            ON CREATE SET dp.updated_at = datetime($updated_at)
            ON MATCH SET dp.updated_at = datetime($updated_at)
            WITH dp
            MATCH (t:Task {id: $task_id})
            OPTIONAL MATCH (dp)-[inc:INCLUDES]->()
            WITH dp, t, coalesce(max(inc.order) + 1, 0) AS next_order
            WHERE NOT EXISTS { MATCH (dp)-[:INCLUDES]->(t) }
            CREATE (dp)-[:INCLUDES {order: next_order}]->(t)
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("date", date.to_string())
        .param("task_id", task_id.to_string())
        .param("updated_at", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Remove one task from the day plan. No-op when the plan, the task or the
    /// edge does not exist.
    pub async fn remove_from_day(&self, user_id: Uuid, date: &str, task_id: Uuid) -> Result<()> {
        let q = query(
            r#"
            MATCH (dp:DayPlan {user_id: $user_id, date: $date})
            SET dp.updated_at = datetime($updated_at)
            WITH dp
            OPTIONAL MATCH (dp)-[inc:INCLUDES]->(t:Task {id: $task_id})
            DELETE inc
            "#,
        )
        .param("user_id", user_id.to_string())
        .param("date", date.to_string())
        .param("task_id", task_id.to_string())
        .param("updated_at", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }
}
