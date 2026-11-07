//! T7 côté Neo4j : les quatre lectures qui rendent une assertion réfutable.
//!
//! Rien n'est deviné ici : un type de cible inconnu rend `None` (« je ne sais pas »), jamais
//! `false` (« c'est absent »). C'est la même distinction que le verdict à trois états du
//! module `notes::beliefs` — la faire disparaître au niveau du store la rendrait invisible
//! un étage plus haut.

use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

use crate::neo4j::client::Neo4jClient;
use crate::notes::models::{AssertionResult, Note, NoteStatus};

/// Le label et la propriété d'un type de cible. `None` = type inconnu, donc invérifiable.
///
/// Fermé volontairement : ces deux chaînes sont interpolées dans une requête.
fn label_and_prop(kind: &str) -> Option<(&'static str, &'static str)> {
    match kind {
        "file" => Some(("File", "path")),
        "function" => Some(("Function", "name")),
        "struct" => Some(("Struct", "name")),
        "trait" => Some(("Trait", "name")),
        "enum" => Some(("Enum", "name")),
        "impl" => Some(("Impl", "name")),
        _ => None,
    }
}

/// Les relations qu'une assertion peut interroger. Liste fermée : `edge` est interpolé.
const LINK_EDGES: &[&str] = &["CALLS", "IMPLEMENTS_TRAIT", "DEPENDS_ON", "IMPORTS"];

impl Neo4jClient {
    /// See [`crate::neo4j::AnalyticsStore::entity_exists`].
    pub async fn entity_exists(&self, kind: &str, name: &str) -> Result<Option<bool>> {
        let Some((label, prop)) = label_and_prop(kind) else {
            return Ok(None);
        };
        let statement =
            format!("MATCH (n:{label}) WHERE n.{prop} = $name RETURN count(n) > 0 AS ok");
        let mut rows = self
            .graph
            .execute(query(&statement).param("name", name.to_string()))
            .await?;
        Ok(rows
            .next()
            .await?
            .map(|row| row.get::<bool>("ok").unwrap_or(false)))
    }

    /// See [`crate::neo4j::AnalyticsStore::entities_linked`].
    pub async fn entities_linked(
        &self,
        from_kind: &str,
        from_name: &str,
        edge: &str,
        to_kind: &str,
        to_name: &str,
    ) -> Result<Option<bool>> {
        if !LINK_EDGES.contains(&edge) {
            // Une relation inconnue n'est pas une réfutation : c'est une question qu'on ne
            // sait pas poser au graphe. Lever une erreur ici ferait échouer TOUTE la passe
            // de re-vérification à cause d'une seule assertion mal écrite — une croyance
            // fautive rendrait muettes toutes les autres. `None` = « je ne sais pas ».
            return Ok(None);
        }
        let (Some((from_label, from_prop)), Some((to_label, to_prop))) =
            (label_and_prop(from_kind), label_and_prop(to_kind))
        else {
            return Ok(None);
        };
        let statement = format!(
            "MATCH (a:{from_label})-[r:{edge}]->(b:{to_label}) \
             WHERE a.{from_prop} = $from AND b.{to_prop} = $to \
             RETURN count(r) > 0 AS ok"
        );
        let mut rows = self
            .graph
            .execute(
                query(&statement)
                    .param("from", from_name.to_string())
                    .param("to", to_name.to_string()),
            )
            .await?;
        Ok(rows
            .next()
            .await?
            .map(|row| row.get::<bool>("ok").unwrap_or(false)))
    }

    /// See [`crate::neo4j::AnalyticsStore::list_assertion_notes`].
    pub async fn list_assertion_notes(&self, limit: usize) -> Result<Vec<Note>> {
        let mut rows = self
            .graph
            .execute(
                query(
                    "MATCH (n:Note) WHERE n.assertion_rule_json IS NOT NULL \
                     RETURN n ORDER BY n.created_at ASC LIMIT $limit",
                )
                .param("limit", limit as i64),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let node: neo4rs::Node = row.get("n")?;
            out.push(self.node_to_note(&node)?);
        }
        Ok(out)
    }

    /// See [`crate::neo4j::AnalyticsStore::record_assertion_check`].
    pub async fn record_assertion_check(
        &self,
        note_id: Uuid,
        result: &AssertionResult,
        status: Option<NoteStatus>,
        reason: Option<&str>,
    ) -> Result<()> {
        let payload = serde_json::to_string(result)?;
        // `status` à `None` n'écrit rien : une réfutation `Warn` trace sans casser la note.
        // `reason` va SUR LE NŒUD (et non dans un journal) : c'est ce qui permet à l'agent
        // de relire, plus tard, pourquoi sa croyance a été révisée.
        self.graph
            .run(
                query(
                    "MATCH (n:Note {id: $id}) \
                     SET n.last_assertion_result_json = $result, n.last_checked_at = datetime() \
                     FOREACH (_ IN CASE WHEN $status IS NULL THEN [] ELSE [1] END | \
                        SET n.status = $status, \
                            n.revision_reason = $reason, \
                            n.revised_at = datetime())",
                )
                .param("id", note_id.to_string())
                .param("result", payload)
                .param("status", status.map(|s| s.to_string()))
                .param("reason", reason.map(|r| r.to_string())),
            )
            .await?;
        Ok(())
    }
}
