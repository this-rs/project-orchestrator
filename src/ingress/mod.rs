//! Ingestion d'événements externes (webhooks entrants).
//!
//! Une livraison reçue sur `POST /hooks/events/{source}/{trigger_id}` est vérifiée,
//! puis journalisée SI elle est authentifiée : les refus qui précèdent
//! l'authentification ne sont comptés qu'en mémoire (`guard`). Ce module ne
//! contient aucun secret (la règle ne porte qu'une référence `vault:<nom>`) et ne
//! journalise jamais le corps de la requête.
use crate::neo4j::{
    AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
    ReleaseStore, WorkspaceStore,
};

pub mod guard;
pub mod model;
pub mod rule;

pub use guard::{IngressConfig, IngressRuntime};
pub use model::{ExternalEvent, IngressOutcome};

/// Durée de conservation du journal `ExternalEvent` (purge horaire).
pub const EVENT_RETENTION_DAYS: i64 = 30;

/// Purge les événements externes plus vieux que [`EVENT_RETENTION_DAYS`] (appelée
/// à chaque tour du planificateur horaire, `protocol::hooks::spawn_protocol_scheduler`).
pub async fn purge_expired(
    store: &dyn crate::neo4j::GraphStore,
    now: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<u64> {
    store
        .purge_external_events_before(now - chrono::Duration::days(EVENT_RETENTION_DAYS))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::GraphStore;
    use crate::neo4j::{
        AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
        ReleaseStore, WorkspaceStore,
    };

    fn event(trigger_id: uuid::Uuid, received_at: chrono::DateTime<chrono::Utc>) -> ExternalEvent {
        ExternalEvent {
            id: uuid::Uuid::new_v4(),
            trigger_id,
            source: "github".into(),
            outcome: IngressOutcome::Accepted,
            reason: None,
            delivery_id: None,
            event_type: None,
            action: None,
            body_bytes: 1,
            received_at,
        }
    }

    #[tokio::test]
    async fn the_retention_keeps_thirty_days() {
        let store = MockGraphStore::new();
        let now = chrono::Utc::now();
        let t = uuid::Uuid::new_v4();
        let old = event(t, now - chrono::Duration::days(31));
        let recent = event(t, now - chrono::Duration::days(29));
        store.record_external_event(&old).await.unwrap();
        store.record_external_event(&recent).await.unwrap();
        assert_eq!(purge_expired(&store, now).await.unwrap(), 1);
        let left = store.list_external_events(t, 10).await.unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, recent.id);
    }
}
