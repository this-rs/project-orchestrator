//! `SharingStore` on obrain — PARTIEL.
//!
//! Porté : [`create_sharing_event`](SharingStore::create_sharing_event), l'écriture
//! de la piste d'audit. Un événement est UN nœud `SharingEvent` portant UNE
//! propriété, sa forme JSON (`SharingEvent` est sérialisable) : une seule écriture
//! de propriété, donc pas d'état à moitié écrit au sens du WAL. Une valeur de plus
//! de [`SEUIL_COLONNE_BLOB`] octets va en colonne blob : le flush complet est alors
//! fait avant l'acquittement (`flush_complet`, comme `project.rs`).
//!
//! Refusé NOMMÉMENT (jamais de faux succès, jamais de repli sur Neo4j) :
//!
//! - **Politique de partage** (`get/update_sharing_policy`) : la politique est une
//!   propriété du nœud `Project`. Le passage projet → nœud est l'index privé de
//!   `project.rs`, qu'on ne touche pas dans ce portage.
//! - **Consentements** (`get/update_sharing_consent`,
//!   `update_decision_sharing_consent`, `count_notes_by_consent`) : `Note` et
//!   `Decision` ne sont pas portés sur cette branche.
//! - **Historique** (`list_sharing_events`) : la référence Neo4j ignore
//!   `project_id` et renvoie les événements de TOUS les projets, et le handler
//!   (`api/sharing_handlers.rs::get_sharing_history`) ne vérifie pas le
//!   propriétaire. Un `SharingEvent` ne porte aucun lien vers son projet : le
//!   reproduire exposerait les données d'autrui. Refus jusqu'à ce que l'événement
//!   soit rattaché à son projet et que le contrôle d'accès soit posé.
//! - **Tombstones** (`persist_tombstone`, `list_tombstones`, `is_tombstoned`) :
//!   la persistance est un MERGE sur `content_hash`, et la lecture de
//!   `is_tombstoned` est sur le chemin chaud ; il faut un index sur `content_hash`
//!   qui n'existe pas encore.

use anyhow::Result;
use async_trait::async_trait;
use obrain_common::types::Value;
use uuid::Uuid;

use super::{Ecriture, NonPorte, ObrainGraphStore, SEUIL_COLONNE_BLOB};
use crate::episodes::distill_models::{ConsentCounts, SharingConsent, SharingEvent, SharingPolicy};
use crate::neo4j::traits::SharingStore;
use crate::reception::anchor::SignedTombstone;

const LABEL_EVENEMENT: &str = "SharingEvent";
const PROPRIETE_EVENEMENT: &str = "evenement";

/// Refus nommé : le nom de la méthode (`NonPorte`, downcastable) et la raison.
fn non_porte<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

#[async_trait]
impl SharingStore for ObrainGraphStore {
    async fn get_sharing_policy(&self, _project_id: Uuid) -> Result<Option<SharingPolicy>> {
        non_porte(
            "SharingStore::get_sharing_policy",
            "la politique est une propriété du nœud Project ; l'index projet → nœud est \
             privé de project.rs, non touché par ce portage",
        )
    }

    async fn update_sharing_policy(&self, _project_id: Uuid, _policy: &SharingPolicy) -> Result<()> {
        non_porte(
            "SharingStore::update_sharing_policy",
            "la politique est une propriété du nœud Project ; l'index projet → nœud est \
             privé de project.rs, non touché par ce portage",
        )
    }

    async fn get_sharing_consent(&self, _note_id: Uuid) -> Result<SharingConsent> {
        non_porte(
            "SharingStore::get_sharing_consent",
            "le domaine Note n'est pas porté sur cette branche",
        )
    }

    async fn update_sharing_consent(&self, _note_id: Uuid, _consent: &SharingConsent) -> Result<()> {
        non_porte(
            "SharingStore::update_sharing_consent",
            "le domaine Note n'est pas porté sur cette branche",
        )
    }

    async fn update_decision_sharing_consent(
        &self,
        _decision_id: Uuid,
        _consent: &SharingConsent,
    ) -> Result<()> {
        non_porte(
            "SharingStore::update_decision_sharing_consent",
            "le domaine Decision n'est pas porté sur cette branche",
        )
    }

    async fn count_notes_by_consent(&self, _project_id: Option<Uuid>) -> Result<ConsentCounts> {
        non_porte(
            "SharingStore::count_notes_by_consent",
            "le domaine Note n'est pas porté sur cette branche",
        )
    }

    async fn create_sharing_event(&self, event: &SharingEvent) -> Result<()> {
        let json = serde_json::to_string(event)?;
        let flush_complet = json.len() > SEUIL_COLONNE_BLOB;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                w.flush_complet = flush_complet;
                let noeud = w.db.create_node(&[LABEL_EVENEMENT]);
                w.db.set_node_property(noeud, PROPRIETE_EVENEMENT, Value::from(json));
                Ok(())
            },
        )
        .await
    }

    async fn list_sharing_events(
        &self,
        _project_id: Uuid,
        _limit: i64,
        _offset: i64,
    ) -> Result<Vec<SharingEvent>> {
        non_porte(
            "SharingStore::list_sharing_events",
            "un SharingEvent ne porte aucun lien vers son projet : la référence Neo4j renvoie \
             les événements de tous les projets (project_id ignoré). Refus plutôt que fuite",
        )
    }

    async fn persist_tombstone(&self, _tombstone: &SignedTombstone) -> Result<()> {
        non_porte(
            "SharingStore::persist_tombstone",
            "MERGE sur content_hash : il faut un index sur content_hash, non porté",
        )
    }

    async fn list_tombstones(&self) -> Result<Vec<SignedTombstone>> {
        non_porte(
            "SharingStore::list_tombstones",
            "les tombstones ne sont pas portés (index content_hash absent)",
        )
    }

    async fn is_tombstoned(&self, _content_hash: &str) -> Result<bool> {
        non_porte(
            "SharingStore::is_tombstoned",
            "les tombstones ne sont pas portés (index content_hash absent, chemin chaud)",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::distill_models::PrivacyMode;
    use crate::obrain::{methodes_portees, methodes_refusees};

    fn un_evenement(reason: Option<String>) -> SharingEvent {
        SharingEvent {
            id: Uuid::new_v4().to_string(),
            content_hash: "hash".into(),
            artifact_type: "note".into(),
            action: "shared".into(),
            source_did: "did:local:a".into(),
            target_did: "broadcast".into(),
            timestamp: chrono::Utc::now(),
            consent: SharingConsent::default(),
            privacy_mode: PrivacyMode::default(),
            reason,
        }
    }

    #[tokio::test]
    async fn creation_d_evenement_est_acceptee_et_longue_valeur_aussi() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        store.create_sharing_event(&un_evenement(None)).await.unwrap();
        // Une raison au-delà du seuil de colonne blob : flush complet, toujours Ok.
        let long = "x".repeat(SEUIL_COLONNE_BLOB * 2);
        store
            .create_sharing_event(&un_evenement(Some(long)))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn toutes_les_lectures_sont_refusees_par_nom() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let id = Uuid::new_v4();
        let err = store.list_sharing_events(id, 10, 0).await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("SharingStore::list_sharing_events"))
        );
        let err = store.is_tombstoned("h").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("SharingStore::is_tombstoned"))
        );
    }

    #[test]
    fn le_manifeste_couvre_toutes_les_methodes_de_sharing() {
        // Écrites à la main (portées ou refus nommés) : listées dans PORTEES.
        assert!(methodes_portees().contains(&"SharingStore::create_sharing_event"));
        assert!(methodes_portees().contains(&"SharingStore::list_sharing_events"));
        // Aucune méthode de SharingStore n'est laissée à la génération.
        assert!(!methodes_refusees()
            .iter()
            .any(|m| m.starts_with("SharingStore::")));
    }
}
