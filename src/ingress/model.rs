//! Modèle du journal d'événements externes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Issue d'une livraison reçue.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngressOutcome {
    /// Vérifiée et retenue par le déclencheur.
    Accepted,
    /// Refusée ; `reason` dit pourquoi (code stable, jamais de secret).
    Rejected,
}

/// Une livraison reçue pour un déclencheur, telle qu'elle est journalisée.
///
/// Le corps n'est pas conservé : seuls sa taille et les en-têtes d'identification
/// (livraison, type, action) le sont.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalEvent {
    pub id: Uuid,
    /// Déclencheur visé par l'URL.
    pub trigger_id: Uuid,
    /// Source déclarée dans l'URL (`github`, ou une source générique).
    pub source: String,
    pub outcome: IngressOutcome,
    /// Code stable de la raison (`invalid_signature`, `replayed_delivery`, ...).
    pub reason: Option<String>,
    /// Identifiant de livraison fourni par l'émetteur (`X-GitHub-Delivery`).
    pub delivery_id: Option<String>,
    /// Type d'événement (`X-GitHub-Event`).
    pub event_type: Option<String>,
    /// Action de la charge utile (`action`), quand elle existe.
    pub action: Option<String>,
    /// Taille du corps reçu, en octets.
    pub body_bytes: u64,
    pub received_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::neo4j::{
        AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
        ReleaseStore, WorkspaceStore,
    };
    use chrono::TimeZone;

    fn event(trigger_id: Uuid, outcome: IngressOutcome, secs: i64) -> ExternalEvent {
        ExternalEvent {
            id: Uuid::new_v4(),
            trigger_id,
            source: "github".into(),
            outcome,
            reason: None,
            delivery_id: Some(format!("d-{secs}")),
            event_type: Some("push".into()),
            action: None,
            body_bytes: 42,
            received_at: Utc.timestamp_opt(1_800_000_000 + secs, 0).unwrap(),
        }
    }

    #[tokio::test]
    async fn journal_lists_newest_first_for_one_trigger_only() {
        let store = MockGraphStore::new();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        store
            .record_external_event(&event(a, IngressOutcome::Accepted, 1))
            .await
            .unwrap();
        store
            .record_external_event(&event(a, IngressOutcome::Rejected, 3))
            .await
            .unwrap();
        store
            .record_external_event(&event(b, IngressOutcome::Accepted, 2))
            .await
            .unwrap();

        let listed = store.list_external_events(a, 10).await.unwrap();
        assert_eq!(
            listed.len(),
            2,
            "trigger b's event must not leak into a's journal"
        );
        assert_eq!(listed[0].outcome, IngressOutcome::Rejected);
        assert_eq!(listed[1].outcome, IngressOutcome::Accepted);
    }

    #[tokio::test]
    async fn journal_honours_the_limit() {
        let store = MockGraphStore::new();
        let a = Uuid::new_v4();
        for s in 0..5 {
            store
                .record_external_event(&event(a, IngressOutcome::Accepted, s))
                .await
                .unwrap();
        }
        assert_eq!(store.list_external_events(a, 2).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_delivery_is_claimed_once_per_trigger() {
        let store = MockGraphStore::new();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(store
            .claim_external_delivery(a, "github:d-1")
            .await
            .unwrap());
        assert!(
            !store
                .claim_external_delivery(a, "github:d-1")
                .await
                .unwrap(),
            "replay refused"
        );
        assert!(
            store
                .claim_external_delivery(b, "github:d-1")
                .await
                .unwrap(),
            "per trigger"
        );
    }
}
