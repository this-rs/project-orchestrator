//! Journal des événements externes (`:ExternalEvent`) et réservation des livraisons
//! (`:ExternalDelivery`), sur le même modèle que les firings de déclencheurs.

use super::client::Neo4jClient;
use crate::ingress::{ExternalEvent, IngressOutcome};
use anyhow::Result;
use chrono::SecondsFormat;
use neo4rs::query;
use uuid::Uuid;

fn outcome_str(outcome: IngressOutcome) -> &'static str {
    match outcome {
        IngressOutcome::Accepted => "accepted",
        IngressOutcome::Rejected => "rejected",
    }
}

fn parse_outcome(raw: &str) -> Result<IngressOutcome> {
    match raw {
        "accepted" => Ok(IngressOutcome::Accepted),
        "rejected" => Ok(IngressOutcome::Rejected),
        other => anyhow::bail!("unknown ingress outcome: {other}"),
    }
}

impl Neo4jClient {
    pub async fn record_external_event_impl(&self, event: &ExternalEvent) -> Result<()> {
        let q = query(
            r#"
            CREATE (e:ExternalEvent {
                id: $id,
                trigger_id: $trigger_id,
                source: $source,
                outcome: $outcome,
                reason: $reason,
                delivery_id: $delivery_id,
                event_type: $event_type,
                action: $action,
                body_bytes: $body_bytes,
                received_at: $received_at
            })
            "#,
        )
        .param("id", event.id.to_string())
        .param("trigger_id", event.trigger_id.to_string())
        .param("source", event.source.clone())
        .param("outcome", outcome_str(event.outcome))
        .param("reason", event.reason.clone().unwrap_or_default())
        .param("delivery_id", event.delivery_id.clone().unwrap_or_default())
        .param("event_type", event.event_type.clone().unwrap_or_default())
        .param("action", event.action.clone().unwrap_or_default())
        .param("body_bytes", event.body_bytes as i64)
        // Largeur fixe : l'ordre lexicographique est l'ordre chronologique.
        .param(
            "received_at",
            event
                .received_at
                .to_rfc3339_opts(SecondsFormat::Nanos, true),
        );
        self.graph.run(q).await?;
        Ok(())
    }

    pub async fn list_external_events_impl(
        &self,
        trigger_id: Uuid,
        limit: i64,
    ) -> Result<Vec<ExternalEvent>> {
        let q = query(
            r#"
            MATCH (e:ExternalEvent {trigger_id: $trigger_id})
            RETURN e
            ORDER BY e.received_at DESC
            LIMIT $limit
            "#,
        )
        .param("trigger_id", trigger_id.to_string())
        .param("limit", limit);

        let opt = |node: &neo4rs::Node, key: &str| -> Option<String> {
            node.get::<String>(key).ok().filter(|s| !s.is_empty())
        };
        let mut result = self.graph.execute(q).await?;
        let mut events = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("e")?;
            let outcome: String = node.get("outcome")?;
            let body_bytes: i64 = node.get("body_bytes")?;
            let received_at: String = node.get("received_at")?;
            events.push(ExternalEvent {
                id: node.get::<String>("id")?.parse()?,
                trigger_id: node.get::<String>("trigger_id")?.parse()?,
                source: node.get("source")?,
                outcome: parse_outcome(&outcome)?,
                reason: opt(&node, "reason"),
                delivery_id: opt(&node, "delivery_id"),
                event_type: opt(&node, "event_type"),
                action: opt(&node, "action"),
                body_bytes: body_bytes.max(0) as u64,
                received_at: received_at.parse()?,
            });
        }
        Ok(events)
    }

    /// Réserve atomiquement la livraison `delivery_key` du déclencheur : `true` pour
    /// le premier appelant seulement, `false` pour un rejeu.
    pub async fn claim_external_delivery_impl(
        &self,
        trigger_id: Uuid,
        delivery_key: &str,
    ) -> Result<bool> {
        let now = chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true);
        let q = query(
            r#"
            MERGE (d:ExternalDelivery {trigger_id: $trigger_id, delivery_key: $delivery_key})
            ON CREATE SET d.claimed_at = $now
            RETURN d.claimed_at = $now AS created
            "#,
        )
        .param("trigger_id", trigger_id.to_string())
        .param("delivery_key", delivery_key.to_string())
        .param("now", now);

        let mut result = self.graph.execute(q).await?;
        let row = result
            .next()
            .await?
            .ok_or_else(|| anyhow::anyhow!("claim_external_delivery returned no row"))?;
        Ok(row.get::<bool>("created")?)
    }
}
