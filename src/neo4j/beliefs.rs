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

/// Le label et la propriété qui porte le nom du nœud.
///
/// `None` = type non interrogeable par nom, donc invérifiable. Fermé volontairement :
/// ces chaînes sont interpolées dans une requête.
///
/// `impl` est ABSENT de cette liste, et c'est un correctif : un `Impl` ne porte pas de
/// propriété `name` (il a `for_type`, `trait_name`, `stable_key`). L'ancienne table
/// répondait donc `false` à « cette impl existe » — un faux « absent », c'est-à-dire une
/// croyance réfutée à tort. Un type qu'on ne sait pas nommer devient **invérifiable**,
/// ce que le module distingue déjà partout ailleurs.
struct Target {
    label: &'static str,
    name_prop: &'static str,
}

fn target_of(kind: &str) -> Option<Target> {
    let (label, name_prop) = match kind {
        "file" => ("File", "path"),
        "function" => ("Function", "name"),
        "struct" => ("Struct", "name"),
        "trait" => ("Trait", "name"),
        "enum" => ("Enum", "name"),
        _ => return None,
    };
    Some(Target { label, name_prop })
}

/// La clause qui borne `alias` au projet demandé, ou ne borne rien quand il n'y en a pas.
///
/// Sans elle, `Function.name` est ambigu par nature : deux projets peuvent avoir une
/// fonction du même nom, et une croyance se vérifierait contre l'homonyme de l'autre.
///
/// Pas de variante par label : **tous** les types nommables portent `project_id`, et
/// chacun a son index `*_project_id` (function, struct, trait, enum, file). Un label qui
/// n'en porterait pas n'a rien à faire dans [`target_of`] — `impl` en est la preuve.
fn within_project(alias: &str, target: &Target) -> String {
    debug_assert!(
        !target.label.is_empty(),
        "un label interpolé n'est jamais vide"
    );
    format!("($project IS NULL OR {alias}.project_id = $project)")
}

/// Les relations qu'une assertion peut interroger. Liste fermée : `edge` est interpolé.
const LINK_EDGES: &[&str] = &["CALLS", "IMPLEMENTS_TRAIT", "DEPENDS_ON", "IMPORTS"];

impl Neo4jClient {
    /// See [`crate::neo4j::AnalyticsStore::entity_exists`].
    pub async fn entity_exists(
        &self,
        kind: &str,
        name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Option<bool>> {
        let Some(target) = target_of(kind) else {
            return Ok(None);
        };
        let statement = format!(
            "MATCH (n:{label}) WHERE n.{prop} = $name AND {scope} RETURN count(n) > 0 AS ok",
            label = target.label,
            prop = target.name_prop,
            scope = within_project("n", &target)
        );
        let mut rows = self
            .graph
            .execute(
                query(&statement)
                    .param("name", name.to_string())
                    .param("project", project_id.map(|p| p.to_string())),
            )
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
        project_id: Option<Uuid>,
    ) -> Result<Option<bool>> {
        if !LINK_EDGES.contains(&edge) {
            // Une relation inconnue n'est pas une réfutation : c'est une question qu'on ne
            // sait pas poser au graphe. Lever une erreur ici ferait échouer TOUTE la passe
            // de re-vérification à cause d'une seule assertion mal écrite — une croyance
            // fautive rendrait muettes toutes les autres. `None` = « je ne sais pas ».
            return Ok(None);
        }
        let (Some(from), Some(to)) = (target_of(from_kind), target_of(to_kind)) else {
            return Ok(None);
        };
        let statement = format!(
            "MATCH (a:{from_label})-[r:{edge}]->(b:{to_label}) \
             WHERE a.{from_prop} = $from AND b.{to_prop} = $to \
               AND {from_scope} AND {to_scope} \
             RETURN count(r) > 0 AS ok",
            from_label = from.label,
            to_label = to.label,
            from_prop = from.name_prop,
            to_prop = to.name_prop,
            from_scope = within_project("a", &from),
            to_scope = within_project("b", &to)
        );
        let mut rows = self
            .graph
            .execute(
                query(&statement)
                    .param("from", from_name.to_string())
                    .param("to", to_name.to_string())
                    .param("project", project_id.map(|p| p.to_string())),
            )
            .await?;
        Ok(rows
            .next()
            .await?
            .map(|row| row.get::<bool>("ok").unwrap_or(false)))
    }

    /// See [`crate::neo4j::AnalyticsStore::project_synced`].
    pub async fn project_synced(&self, project_id: Uuid) -> Result<Option<bool>> {
        let mut rows = self
            .graph
            .execute(
                query("MATCH (p:Project {id: $id}) RETURN p.last_synced IS NOT NULL AS synced")
                    .param("id", project_id.to_string()),
            )
            .await?;
        Ok(rows
            .next()
            .await?
            .map(|row| row.get::<bool>("synced").unwrap_or(false)))
    }

    /// See [`crate::neo4j::AnalyticsStore::list_assertion_notes`].
    ///
    /// Le prédicat n'est pas `IS NOT NULL` : `create_note` écrit `assertion_rule_json` sur
    /// **toutes** les notes, avec la chaîne VIDE quand il n'y a pas de règle. `IS NOT NULL`
    /// matchait donc tout le corpus — mesuré le 07/11 sur l'instance vivante : **30 813**
    /// notes, dont **zéro** portant une vraie règle. La re-vérification examinait les 200
    /// plus anciennes (aucune croyance), publiait `unverifiable: 200`, et n'aurait jamais
    /// atteint une croyance créée après elles. Le mock, lui, filtrait sur la règle parsée :
    /// d'où un écart mock/réel qu'aucun test ne voyait.
    ///
    /// L'ordre fait aussi le TOUR des croyances : `last_checked_at` d'abord (les jamais
    /// vérifiées, puis les plus anciennement vérifiées), sinon `created_at`. Sans ça, une
    /// limite de 200 figerait éternellement le même sous-ensemble.
    pub async fn list_assertion_notes(&self, limit: usize) -> Result<Vec<Note>> {
        let mut rows = self
            .graph
            .execute(
                query(
                    "MATCH (n:Note) \
                     WHERE n.assertion_rule_json IS NOT NULL \
                       AND n.assertion_rule_json <> '' \
                       AND n.assertion_rule_json <> 'null' \
                     RETURN n \
                     ORDER BY coalesce(n.last_checked_at, n.created_at) ASC LIMIT $limit",
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
