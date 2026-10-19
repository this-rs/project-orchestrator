//! Ingestion d'événements externes (webhooks entrants).
//!
//! Une livraison reçue sur `POST /hooks/events/{source}/{trigger_id}` est vérifiée,
//! puis journalisée : chaque événement reçu, accepté ou rejeté, laisse une trace
//! consultable (`ExternalEvent`). Ce module ne contient aucun secret et ne journalise
//! jamais le corps de la requête.

pub mod model;

pub use model::{ExternalEvent, IngressOutcome};
