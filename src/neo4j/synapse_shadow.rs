//! Neo4j side of the synaptic layer in shadow mode
//! (`crate::neurons::synaptic_shadow`).
//!
//! Every query here reads or writes ONLY the `shadow_*` properties of existing
//! `SYNAPSE` edges between two notes of one project, and `SynapseShadowTrace`
//! nodes. `SYNAPSE.weight` is read (as the starting value of a shadow weight)
//! but never written; no edge is ever created or deleted.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

use super::client::Neo4jClient;
use crate::neurons::synaptic_shadow::{
    estimate_tokens, ShadowDecayScope, ShadowEdgeRow, ShadowNode, ShadowTraceFilter,
    SynapseShadowTrace, TaskClosure, CLOSURE_KEY,
};

fn parse_at(s: Option<String>) -> Option<DateTime<Utc>> {
    s.and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|d| d.with_timezone(&Utc))
}

fn trace_from_row(row: &neo4rs::Row) -> Result<SynapseShadowTrace> {
    let payload: String = row.get("payload").context("trace payload")?;
    let mut trace: SynapseShadowTrace =
        serde_json::from_str(&payload).context("trace payload is not a trace")?;
    // The claimed rewards live on the node (written after the payload).
    trace.rewarded = row.get::<Vec<String>>("rewarded").unwrap_or_default();
    Ok(trace)
}

impl Neo4jClient {
    pub async fn shadow_synapse_edges(
        &self,
        project_id: Uuid,
        from: &[Uuid],
    ) -> Result<Vec<ShadowEdgeRow>> {
        if from.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<String> = from.iter().map(Uuid::to_string).collect();
        let mut result = self
            .graph
            .execute(
                query(
                    r#"
                    UNWIND $ids AS id
                    MATCH (a:Note {id: id})-[s:SYNAPSE]->(b:Note)
                    WHERE a.project_id = $project_id AND b.project_id = $project_id
                      AND a.id <> b.id
                    RETURN a.id AS from, b.id AS to,
                           toFloat(coalesce(s.shadow_weight, s.weight, 0.0)) AS weight,
                           toFloat(coalesce(b.energy, 1.0)) AS energy,
                           toFloat(coalesce(b.scar_intensity, 0.0)) AS scar,
                           coalesce(b.status, 'active') AS status,
                           size(coalesce(b.content, '')) AS chars,
                           toString(b.created_at) AS created_at
                    "#,
                )
                .param("ids", ids)
                .param("project_id", project_id.to_string()),
            )
            .await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            let (Ok(from), Ok(to)) = (
                row.get::<String>("from")?.parse::<Uuid>(),
                row.get::<String>("to")?.parse::<Uuid>(),
            ) else {
                continue;
            };
            let status: String = row.get("status").unwrap_or_default();
            rows.push(ShadowEdgeRow {
                from,
                to,
                weight: row.get("weight").unwrap_or(0.0),
                target: ShadowNode {
                    energy: row.get("energy").unwrap_or(1.0),
                    scar_intensity: row.get("scar").unwrap_or(0.0),
                    reachable: matches!(status.as_str(), "active" | "needs_review"),
                    tokens: estimate_tokens(row.get::<i64>("chars").unwrap_or(0).max(0) as usize),
                    created_at: parse_at(row.get("created_at").ok()),
                },
            });
        }
        Ok(rows)
    }

    pub async fn reward_shadow_synapses(
        &self,
        project_id: Uuid,
        pairs: &[(Uuid, Uuid)],
        delta: f64,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        use super::batch::{bolt_map, BoltMap};
        let mut unique: Vec<(Uuid, Uuid)> = pairs
            .iter()
            .filter(|(a, b)| a != b)
            .map(|(a, b)| if a < b { (*a, *b) } else { (*b, *a) })
            .collect();
        unique.sort();
        unique.dedup();
        let items: Vec<BoltMap> = unique
            .iter()
            .map(|(a, b)| bolt_map(&[("a", a.to_string().into()), ("b", b.to_string().into())]))
            .collect();
        if items.is_empty() {
            return Ok(0);
        }
        let mut written = 0usize;
        for chunk in items.chunks(super::batch::BATCH_SIZE) {
            let mut result = self
                .graph
                .execute(
                    query(
                        r#"
                        UNWIND $items AS pair
                        MATCH (a:Note {id: pair.a})-[s:SYNAPSE]-(b:Note {id: pair.b})
                        WHERE a.project_id = $project_id AND b.project_id = $project_id
                        WITH s, toFloat(coalesce(s.shadow_weight, s.weight, 0.0)) + $delta AS w
                        SET s.shadow_weight = CASE WHEN w > 1.0 THEN 1.0 WHEN w < 0.0 THEN 0.0 ELSE w END,
                            s.shadow_rewarded_at = datetime($now),
                            s.shadow_reward_count = coalesce(s.shadow_reward_count, 0) + 1,
                            s.shadow_decayed_at = coalesce(s.shadow_decayed_at, datetime($now))
                        RETURN count(s) AS written
                        "#,
                    )
                    .param("items", chunk.to_vec())
                    .param("project_id", project_id.to_string())
                    .param("delta", delta)
                    .param("now", now.to_rfc3339()),
                )
                .await?;
            if let Some(row) = result.next().await? {
                written += row.get::<i64>("written").unwrap_or(0) as usize;
            }
        }
        Ok(written)
    }

    pub async fn decay_shadow_synapses(
        &self,
        scope: &ShadowDecayScope,
        half_life_days: f64,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        if !scope.default_enabled && scope.enabled.is_empty() {
            return Ok(0);
        }
        let dormant = self.dormant_project_ids().await?;
        let ids = |v: &[Uuid]| v.iter().map(Uuid::to_string).collect::<Vec<_>>();
        let mut result = self
            .graph
            .execute(
                query(
                    r#"
                    MATCH (a:Note)-[s:SYNAPSE]->(b:Note)
                    WHERE a.project_id IS NOT NULL AND a.project_id <> '' AND a.project_id = b.project_id
                      AND NOT a.project_id IN $disabled
                      AND ($default_enabled OR a.project_id IN $enabled)
                    WITH s, a.project_id IN $dormant AS frozen,
                         toFloat(coalesce(s.shadow_weight, s.weight, 0.0)) AS w0,
                         CASE WHEN s.shadow_decayed_at IS NULL THEN 0.0
                              ELSE toFloat(duration.inSeconds(s.shadow_decayed_at, datetime($now)).seconds)
                         END AS elapsed
                    SET s.shadow_weight = CASE
                            WHEN frozen OR elapsed <= 0.0 THEN w0
                            ELSE w0 * (0.5 ^ (elapsed / $half_life_secs))
                        END,
                        s.shadow_decayed_at = CASE
                            WHEN elapsed < 0.0 THEN s.shadow_decayed_at
                            ELSE datetime($now)
                        END
                    RETURN count(s) AS visited
                    "#,
                )
                .param("disabled", ids(&scope.disabled))
                .param("enabled", ids(&scope.enabled))
                .param("default_enabled", scope.default_enabled)
                .param("dormant", dormant)
                .param("now", now.to_rfc3339())
                .param("half_life_secs", half_life_days * 86_400.0),
            )
            .await?;
        Ok(match result.next().await? {
            Some(row) => row.get::<i64>("visited").unwrap_or(0) as usize,
            None => 0,
        })
    }

    pub async fn record_synapse_shadow_trace(
        &self,
        trace: &SynapseShadowTrace,
        keep: usize,
    ) -> Result<()> {
        let payload = serde_json::to_string(trace)?;
        let note_ids: Vec<String> = trace.note_ids().iter().map(Uuid::to_string).collect();
        self.graph
            .run(
                query(
                    r#"
                    CREATE (:SynapseShadowTrace {
                        id: $id, session_id: $session_id, project_id: $project_id,
                        created_at: datetime($created_at), schema_version: $schema_version,
                        layer_version: $layer_version, note_ids: $note_ids,
                        rewarded: [], payload: $payload
                    })
                    "#,
                )
                .param("id", trace.id.to_string())
                .param("session_id", trace.session_id.to_string())
                .param("project_id", trace.project_id.to_string())
                .param("created_at", trace.created_at.to_rfc3339())
                .param("schema_version", trace.schema_version as i64)
                .param("layer_version", trace.layer_version.clone())
                .param("note_ids", note_ids)
                .param("payload", payload),
            )
            .await?;
        self.graph
            .run(
                query(
                    r#"
                    MATCH (t:SynapseShadowTrace {session_id: $session_id})
                    WITH t ORDER BY t.created_at DESC, t.id DESC
                    SKIP $keep
                    DETACH DELETE t
                    "#,
                )
                .param("session_id", trace.session_id.to_string())
                .param("keep", keep as i64),
            )
            .await?;
        Ok(())
    }

    pub async fn list_synapse_shadow_traces(
        &self,
        filter: &ShadowTraceFilter,
    ) -> Result<Vec<SynapseShadowTrace>> {
        let q = match filter {
            ShadowTraceFilter::Session(session_id) => query(
                r#"
                MATCH (t:SynapseShadowTrace {session_id: $session_id})
                RETURN t.payload AS payload, t.rewarded AS rewarded
                ORDER BY t.created_at, t.id
                "#,
            )
            .param("session_id", session_id.to_string()),
            ShadowTraceFilter::ContainingNote { note_id, limit } => query(
                r#"
                MATCH (t:SynapseShadowTrace)
                WHERE $note_id IN t.note_ids
                RETURN t.payload AS payload, t.rewarded AS rewarded
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT $limit
                "#,
            )
            .param("note_id", note_id.to_string())
            .param("limit", *limit as i64),
        };
        let mut result = self.graph.execute(q).await?;
        let mut traces = Vec::new();
        while let Some(row) = result.next().await? {
            traces.push(trace_from_row(&row)?);
        }
        Ok(traces)
    }

    pub async fn get_synapse_shadow_trace(&self, id: Uuid) -> Result<Option<SynapseShadowTrace>> {
        let mut result = self
            .graph
            .execute(
                query(
                    "MATCH (t:SynapseShadowTrace {id: $id}) \
                     RETURN t.payload AS payload, t.rewarded AS rewarded",
                )
                .param("id", id.to_string()),
            )
            .await?;
        match result.next().await? {
            Some(row) => Ok(Some(trace_from_row(&row)?)),
            None => Ok(None),
        }
    }

    pub async fn claim_shadow_trace_reward(&self, trace_id: Uuid, key: &str) -> Result<bool> {
        // Lock first, decide after (a WHERE evaluated before the write lock would
        // let two concurrent claims both see the key absent).
        let mut result = self
            .graph
            .execute(
                query(
                    r#"
                    MATCH (t:SynapseShadowTrace {id: $id})
                    SET t.claim_lock = true
                    WITH t, NOT $key IN coalesce(t.rewarded, []) AS fresh
                    FOREACH (_ IN CASE WHEN fresh THEN [1] ELSE [] END |
                        SET t.rewarded = coalesce(t.rewarded, []) + $key)
                    REMOVE t.claim_lock
                    RETURN fresh
                    "#,
                )
                .param("id", trace_id.to_string())
                .param("key", key.to_string()),
            )
            .await?;
        Ok(match result.next().await? {
            Some(row) => row.get::<bool>("fresh").unwrap_or(false),
            None => false,
        })
    }

    pub async fn shadow_trace_closures(
        &self,
        cutoff: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<(Uuid, TaskClosure)>> {
        let mut result = self
            .graph
            .execute(
                query(
                    r#"
                    MATCH (t:SynapseShadowTrace)
                    WHERE NOT $key IN coalesce(t.rewarded, [])
                    MATCH (ae:AgentExecution {session_id: t.session_id})
                    WHERE ae.completed_at IS NOT NULL AND ae.completed_at < datetime($cutoff)
                    MATCH (task:Task {id: ae.task_id})
                    OPTIONAL MATCH (later:AgentExecution {task_id: ae.task_id})
                    WHERE later.started_at > ae.started_at
                    WITH t, ae, task, count(later) AS retries
                    WITH t, CASE
                        WHEN retries > 0 THEN 'restarted'
                        WHEN task.status = 'completed' THEN 'quiet'
                        ELSE 'open'
                    END AS one
                    // A session with several attempts: one restart is enough.
                    WITH t, collect(one) AS outcomes
                    WITH t, CASE
                        WHEN 'restarted' IN outcomes THEN 'restarted'
                        WHEN 'quiet' IN outcomes THEN 'quiet'
                        ELSE 'open'
                    END AS closure
                    WHERE closure <> 'open'
                    RETURN t.id AS id, closure, t.created_at AS at
                    ORDER BY at, id
                    LIMIT $limit
                    "#,
                )
                .param("key", CLOSURE_KEY)
                .param("cutoff", cutoff.to_rfc3339())
                .param("limit", limit as i64),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = result.next().await? {
            let Ok(id) = row.get::<String>("id")?.parse::<Uuid>() else {
                continue;
            };
            let closure = match row.get::<String>("closure")?.as_str() {
                "restarted" => TaskClosure::Restarted,
                _ => TaskClosure::Quiet,
            };
            out.push((id, closure));
        }
        Ok(out)
    }
}
