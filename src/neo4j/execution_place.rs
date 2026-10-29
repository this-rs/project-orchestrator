//! Neo4j storage of the execution-place audit (T5): `ExecutionPlaceEvent` nodes, append-only,
//! linked to their task by `(Task)-[:HAS_PLACE_EVENT]->(ExecutionPlaceEvent)` when the task exists.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::client::Neo4jClient;
use super::models::{ExecutionPlaceEvent, ExecutionPlaceEventKind};
use anyhow::{Context, Result};
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    /// Append one audit entry. `CREATE` only: an entry is never updated.
    pub async fn create_execution_place_event(&self, event: &ExecutionPlaceEvent) -> Result<()> {
        let q = query(
            r#"
            CREATE (e:ExecutionPlaceEvent {
                id: $id,
                placement_id: $placement_id,
                kind: $kind,
                task_id: $task_id,
                plan_id: $plan_id,
                requested_by: $requested_by,
                owner: $owner,
                actor: $actor,
                session_id: $session_id,
                path: $path,
                repository: $repository,
                base_ref: $base_ref,
                base_sha: $base_sha,
                branch: $branch,
                detail: $detail,
                at: $at
            })
            WITH e
            OPTIONAL MATCH (t:Task {id: $task_id})
            FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |
                CREATE (t)-[:HAS_PLACE_EVENT]->(e))
            "#,
        )
        .param("id", event.id.to_string())
        .param("placement_id", event.placement_id.to_string())
        .param("kind", event.kind.as_str())
        .param("task_id", event.task_id.to_string())
        .param("plan_id", event.plan_id.to_string())
        .param("requested_by", event.requested_by.clone())
        .param("owner", event.owner.clone())
        .param("actor", event.actor.clone().unwrap_or_default())
        .param(
            "session_id",
            event.session_id.map(|s| s.to_string()).unwrap_or_default(),
        )
        .param("path", event.path.clone())
        .param("repository", event.repository.clone())
        .param("base_ref", event.base_ref.clone())
        .param("base_sha", event.base_sha.clone())
        .param("branch", event.branch.clone())
        .param("detail", event.detail.clone().unwrap_or_default())
        .param(
            "at",
            event
                .at
                .to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
        );
        self.graph
            .run(q)
            .await
            .context("create_execution_place_event")?;
        Ok(())
    }

    /// The audit of a task, oldest first.
    pub async fn list_execution_place_events(
        &self,
        task_id: Uuid,
    ) -> Result<Vec<ExecutionPlaceEvent>> {
        let q = query(
            r#"
            MATCH (e:ExecutionPlaceEvent {task_id: $task_id})
            RETURN e
            ORDER BY e.at ASC, e.id ASC
            "#,
        )
        .param("task_id", task_id.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut out = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            out.push(event_from_node(&node)?);
        }
        Ok(out)
    }
}

fn event_from_node(node: &neo4rs::Node) -> Result<ExecutionPlaceEvent> {
    let text = |k: &str| node.get::<String>(k).unwrap_or_default();
    let uuid = |k: &str| {
        Uuid::parse_str(&text(k)).with_context(|| format!("ExecutionPlaceEvent.{k} is not a uuid"))
    };
    let optional = |k: &str| Some(text(k)).filter(|s| !s.is_empty());
    let kind = text("kind");
    Ok(ExecutionPlaceEvent {
        id: uuid("id")?,
        placement_id: uuid("placement_id")?,
        kind: ExecutionPlaceEventKind::parse(&kind)
            .with_context(|| format!("unknown ExecutionPlaceEvent.kind {kind}"))?,
        task_id: uuid("task_id")?,
        plan_id: uuid("plan_id")?,
        requested_by: text("requested_by"),
        owner: text("owner"),
        actor: optional("actor"),
        session_id: optional("session_id").and_then(|s| Uuid::parse_str(&s).ok()),
        path: text("path"),
        repository: text("repository"),
        base_ref: text("base_ref"),
        base_sha: text("base_sha"),
        branch: text("branch"),
        detail: optional("detail"),
        at: chrono::DateTime::parse_from_rfc3339(&text("at"))
            .context("ExecutionPlaceEvent.at")?
            .with_timezone(&chrono::Utc),
    })
}
