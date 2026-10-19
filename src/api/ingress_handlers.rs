//! Réception publique d'événements externes : `POST /hooks/events/{source}/{trigger_id}`.
//!
//! Route PUBLIQUE (hors `require_auth`) : l'accès est gardé par le secret du
//! déclencheur. Ordre des vérifications, chacune refusée avant la suivante :
//! déclencheur connu et de bonne source, déclencheur actif, secret présent,
//! en-tête de signature présent (le corps n'est pas lu s'il manque), taille
//! maximale, signature HMAC-SHA256 valide, livraison non rejouée. Seule une
//! livraison acceptée est journalisée comme telle ; chaque refus est journalisé
//! avec son code de raison. Aucun secret, aucun corps ne sort dans les réponses
//! ni dans les journaux.

use crate::api::handlers::OrchestratorState;
use crate::ingress::{ExternalEvent, IngressOutcome};
use crate::neo4j::GraphStore;
use crate::runner::{Trigger, TriggerType};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use uuid::Uuid;

/// Taille maximale acceptée d'un corps, en octets.
pub const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;

/// En-tête de signature quand le déclencheur n'en désigne pas un autre (format GitHub).
const DEFAULT_SIGNATURE_HEADER: &str = "x-hub-signature-256";

/// Règle d'un déclencheur pour la réception externe, lue de sa configuration.
/// Ne se `Debug` pas : elle porte le secret.
struct Rule {
    source: String,
    secret: Option<String>,
    signature_header: String,
    delivery_header: Option<String>,
}

impl Rule {
    /// Une règle existe seulement si la configuration déclare une `source`.
    fn of(trigger: &Trigger) -> Option<Self> {
        let config = &trigger.config;
        Some(Self {
            source: config.get("source")?.as_str()?.to_string(),
            secret: config
                .get("secret")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            signature_header: config
                .get("signature_header")
                .and_then(|v| v.as_str())
                .unwrap_or(DEFAULT_SIGNATURE_HEADER)
                .to_ascii_lowercase(),
            delivery_header: config
                .get("delivery_header")
                .and_then(|v| v.as_str())
                .map(str::to_ascii_lowercase),
        })
    }
}

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Ce qu'on journalise d'une livraison, quel que soit son sort.
struct Delivery<'a> {
    graph: &'a dyn GraphStore,
    trigger_id: Uuid,
    source: &'a str,
    delivery_id: Option<String>,
    body_bytes: u64,
}

impl Delivery<'_> {
    /// Journalise un refus (best effort : un échec d'écriture est tracé, pas renvoyé)
    /// et rend la réponse générique correspondante.
    async fn refuse(&self, status: StatusCode, reason: &str) -> Response {
        let event = self.event(IngressOutcome::Rejected, Some(reason));
        if let Err(e) = self.graph.record_external_event(&event).await {
            tracing::error!(trigger_id = %self.trigger_id, error = %e, "external event journal write failed");
        }
        tracing::info!(trigger_id = %self.trigger_id, source = self.source, reason, "external event refused");
        (status, Json(serde_json::json!({ "status": "rejected" }))).into_response()
    }

    fn event(&self, outcome: IngressOutcome, reason: Option<&str>) -> ExternalEvent {
        ExternalEvent {
            id: Uuid::new_v4(),
            trigger_id: self.trigger_id,
            source: self.source.to_string(),
            outcome,
            reason: reason.map(str::to_string),
            delivery_id: self.delivery_id.clone(),
            event_type: None,
            action: None,
            body_bytes: self.body_bytes,
            received_at: Utc::now(),
        }
    }
}

/// POST /hooks/events/{source}/{trigger_id} — reçoit une livraison externe.
pub async fn receive_external_event(
    State(state): State<OrchestratorState>,
    Path((source, trigger_id)): Path<(String, Uuid)>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let graph = state.orchestrator.neo4j_arc();

    // Un déclencheur absent, d'un autre type ou d'une autre source est indiscernable :
    // 404 sans journal (on ne journalise pas pour un identifiant qui n'existe pas).
    let trigger = match graph.get_trigger(trigger_id).await {
        Ok(Some(t)) => t,
        Ok(None) => return refuse_unjournaled(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!(error = %e, "external event: trigger lookup failed");
            return refuse_unjournaled(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let rule = match Rule::of(&trigger) {
        Some(rule) if trigger.trigger_type == TriggerType::Webhook && rule.source == source => rule,
        _ => return refuse_unjournaled(StatusCode::NOT_FOUND),
    };

    let declared = headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut delivery = Delivery {
        graph: graph.as_ref(),
        trigger_id,
        source: &source,
        delivery_id: rule
            .delivery_header
            .as_deref()
            .and_then(|h| header_str(&headers, h)),
        body_bytes: declared,
    };

    if !trigger.enabled {
        return delivery
            .refuse(StatusCode::FORBIDDEN, "trigger_disabled")
            .await;
    }
    // Échec fermé : sans secret, aucun déclencheur n'accepte de livraison.
    let Some(secret) = rule.secret.as_deref() else {
        return delivery.refuse(StatusCode::FORBIDDEN, "no_secret").await;
    };
    // Le corps n'est lu qu'à partir d'ici : une livraison sans signature ne coûte rien.
    let Some(signature) = header_str(&headers, &rule.signature_header) else {
        return delivery
            .refuse(StatusCode::UNAUTHORIZED, "missing_signature")
            .await;
    };
    if declared > MAX_BODY_BYTES as u64 {
        return delivery
            .refuse(StatusCode::PAYLOAD_TOO_LARGE, "body_too_large")
            .await;
    }
    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return delivery
                .refuse(StatusCode::PAYLOAD_TOO_LARGE, "body_too_large")
                .await
        }
    };
    delivery.body_bytes = bytes.len() as u64;

    if !crate::runner::providers::webhook::validate_github_signature(secret, &bytes, &signature) {
        return delivery
            .refuse(StatusCode::UNAUTHORIZED, "invalid_signature")
            .await;
    }

    // Identité de la livraison : son en-tête déclaré, sinon le hash du corps.
    let claim_key = match &delivery.delivery_id {
        Some(id) => format!("delivery:{id}"),
        None => format!("body:{}", body_digest(&bytes)),
    };
    match graph.claim_external_delivery(trigger_id, &claim_key).await {
        Ok(true) => {}
        Ok(false) => {
            return delivery
                .refuse(StatusCode::CONFLICT, "replayed_delivery")
                .await
        }
        Err(e) => {
            tracing::error!(trigger_id = %trigger_id, error = %e, "external event: delivery claim failed");
            return refuse_unjournaled(StatusCode::INTERNAL_SERVER_ERROR);
        }
    }

    let event = delivery.event(IngressOutcome::Accepted, None);
    if let Err(e) = graph.record_external_event(&event).await {
        // Une livraison acceptée doit être journalisée : sinon l'émetteur la rejouera.
        tracing::error!(trigger_id = %trigger_id, error = %e, "external event journal write failed");
        return refuse_unjournaled(StatusCode::INTERNAL_SERVER_ERROR);
    }
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "status": "accepted", "event_id": event.id })),
    )
        .into_response()
}

fn refuse_unjournaled(status: StatusCode) -> Response {
    (status, Json(serde_json::json!({ "status": "rejected" }))).into_response()
}

fn body_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::TriggerType;
    use axum::http::Request;
    use tower::ServiceExt;

    /// Secret factice : ce n'est pas un secret réel.
    const SECRET: &str = "fixture-secret-not-real";

    fn sign(body: &[u8]) -> String {
        use hmac::{digest::KeyInit, Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(SECRET.as_bytes()).unwrap();
        mac.update(body);
        format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
    }

    fn trigger(config: serde_json::Value, enabled: bool) -> Trigger {
        Trigger {
            id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            trigger_type: TriggerType::Webhook,
            config,
            enabled,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
            author: None,
            disabled_reason: None,
            approved_mark: None,
        }
    }

    fn secured() -> serde_json::Value {
        serde_json::json!({ "source": "generic", "secret": SECRET, "delivery_header": "x-delivery-id" })
    }

    async fn setup(t: Trigger) -> (OrchestratorState, Uuid) {
        let app = crate::test_helpers::mock_app_state();
        let state = crate::api::handlers::tests::mock_server_state_from(app).await;
        let id = t.id;
        state
            .orchestrator
            .neo4j_arc()
            .create_trigger(&t)
            .await
            .unwrap();
        (state, id)
    }

    /// A body whose reading fails the test: proves the handler never touched it.
    fn unread_body() -> Body {
        Body::from_stream(futures::stream::poll_fn(
            |_| -> std::task::Poll<Option<Result<axum::body::Bytes, std::io::Error>>> {
                panic!("the body was read before the request was judged")
            },
        ))
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        map
    }

    async fn deliver(
        state: &OrchestratorState,
        source: &str,
        id: Uuid,
        pairs: &[(&str, &str)],
        body: Body,
    ) -> StatusCode {
        receive_external_event(
            State(state.clone()),
            Path((source.to_string(), id)),
            headers(pairs),
            body,
        )
        .await
        .status()
    }

    async fn journal(state: &OrchestratorState, id: Uuid) -> Vec<ExternalEvent> {
        state
            .orchestrator
            .neo4j_arc()
            .list_external_events(id, 50)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn refuses_a_trigger_without_secret_and_never_reads_the_body() {
        let (state, id) = setup(trigger(serde_json::json!({ "source": "generic" }), true)).await;
        let status = deliver(
            &state,
            "generic",
            id,
            &[("x-hub-signature-256", "sha256=00")],
            unread_body(),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let events = journal(&state, id).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].outcome, IngressOutcome::Rejected);
        assert_eq!(events[0].reason.as_deref(), Some("no_secret"));
    }

    #[tokio::test]
    async fn a_missing_signature_is_refused_before_the_body_is_read() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let status = deliver(&state, "generic", id, &[], unread_body()).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            journal(&state, id).await[0].reason.as_deref(),
            Some("missing_signature")
        );
    }

    #[tokio::test]
    async fn an_invalid_signature_is_refused_and_journaled() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        let status = deliver(
            &state,
            "generic",
            id,
            &[
                ("x-hub-signature-256", "sha256=00"),
                ("x-delivery-id", "d-1"),
            ],
            Body::from(body.as_slice()),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let events = journal(&state, id).await;
        assert_eq!(events[0].reason.as_deref(), Some("invalid_signature"));
        assert!(events.iter().all(|e| e.outcome == IngressOutcome::Rejected));
    }

    #[tokio::test]
    async fn a_valid_signature_is_accepted_and_journaled_once() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        let sig = sign(body);
        let status = deliver(
            &state,
            "generic",
            id,
            &[("x-hub-signature-256", &sig), ("x-delivery-id", "d-1")],
            Body::from(body.as_slice()),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let events = journal(&state, id).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].outcome, IngressOutcome::Accepted);
        assert_eq!(events[0].delivery_id.as_deref(), Some("d-1"));
        assert_eq!(events[0].body_bytes, body.len() as u64);
    }

    #[tokio::test]
    async fn a_replayed_delivery_is_refused() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        let sig = sign(body);
        let pairs = [
            ("x-hub-signature-256", sig.as_str()),
            ("x-delivery-id", "d-7"),
        ];
        assert_eq!(
            deliver(&state, "generic", id, &pairs, Body::from(body.as_slice())).await,
            StatusCode::ACCEPTED
        );
        assert_eq!(
            deliver(&state, "generic", id, &pairs, Body::from(body.as_slice())).await,
            StatusCode::CONFLICT
        );
        let events = journal(&state, id).await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].reason.as_deref(), Some("replayed_delivery"));
    }

    #[tokio::test]
    async fn an_oversized_body_is_refused_before_it_is_read() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let declared = (MAX_BODY_BYTES + 1).to_string();
        let status = deliver(
            &state,
            "generic",
            id,
            &[
                ("x-hub-signature-256", "sha256=00"),
                ("content-length", &declared),
            ],
            unread_body(),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            journal(&state, id).await[0].reason.as_deref(),
            Some("body_too_large")
        );
    }

    #[tokio::test]
    async fn a_disabled_trigger_refuses_everything() {
        let (state, id) = setup(trigger(secured(), false)).await;
        let status = deliver(
            &state,
            "generic",
            id,
            &[("x-hub-signature-256", "sha256=00")],
            unread_body(),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(
            journal(&state, id).await[0].reason.as_deref(),
            Some("trigger_disabled")
        );
    }

    #[tokio::test]
    async fn unknown_or_foreign_triggers_are_not_found_and_not_journaled() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let status = deliver(&state, "github", id, &[], unread_body()).await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "a source other than the trigger's"
        );
        let status = deliver(&state, "generic", Uuid::new_v4(), &[], unread_body()).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "an unknown trigger");
        assert!(journal(&state, id).await.is_empty());
    }

    #[tokio::test]
    async fn the_route_is_public_and_not_behind_require_auth() {
        let (state, id) = setup(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        let sig = sign(body);
        let app = crate::api::routes::create_router(state.clone());
        let request = Request::builder()
            .method("POST")
            .uri(format!("/hooks/events/generic/{id}"))
            .header("x-hub-signature-256", sig)
            .header("x-delivery-id", "d-public")
            .body(Body::from(body.as_slice()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED, "no JWT was sent");
    }
}
