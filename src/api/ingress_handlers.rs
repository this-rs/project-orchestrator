//! Réception publique d'événements externes : `POST /hooks/events/{source}/{trigger_id}`.
//!
//! Route PUBLIQUE (hors `require_auth`), montée seulement si `ingress.enabled`
//! (éteinte par défaut : 404 comme une route inconnue). L'accès est gardé par le
//! secret du déclencheur, rangé dans le COFFRE (`config.secret_ref = "vault:<nom>"`,
//! grant `Provider("trigger:<id>")`) et résolu à chaque requête par le résolveur des
//! fournisseurs (`VaultCredentialResolver`). Coffre verrouillé ou référence absente :
//! refus neutre, jamais de repli sur une valeur en clair.
//!
//! Ordre des vérifications, chacune refusée avant la suivante :
//! 1. place dans la limite de concurrence (503 immédiat sinon) ;
//! 2. seau GLOBAL de la route (429), avant toute lecture du graphe ;
//! 3. déclencheur connu, `webhook`, de la bonne source (404) ;
//! 4. en-tête de signature présent et bien formé, `sha256=` + 64 hex (401) ;
//! 5. Content-Length au plus `max_body_bytes` (413), AVANT toute lecture ;
//! 6. secret résolu dans le coffre (403 neutre) ;
//! 7. corps lu en flux sous délai (408) et sous plafond (413 dès le dépassement) ;
//! 8. HMAC-SHA256 à temps constant (401) ;
//! 9. seau à jetons du déclencheur (429), débité seulement par un appelant
//!    authentifié : qui ne connaît pas le secret ne peut pas épuiser le quota
//!    d'une livraison légitime.
//!    d'une livraison légitime.
//!
//! Ces refus PRÉCÈDENT l'authentification : ils n'écrivent RIEN en base, ils sont
//! comptés en mémoire (`IngressRuntime::refused`) et tracés en `debug`. Après la
//! signature : déclencheur désactivé (403), rejeu (409, sur le condensat du corps
//! signé ET sur l'identifiant de livraison) et acceptation (202) sont journalisés.
//! Si le journal échoue après la réservation, la réservation est libérée (500) : la
//! relivraison sera acceptée, pas refusée comme rejeu. Aucun secret, aucun corps ne
//! sort dans les réponses ni dans les journaux.

use crate::api::handlers::OrchestratorState;
use crate::ingress::rule::{self, Rule, MIN_SECRET_BYTES};
use crate::ingress::{ExternalEvent, IngressOutcome};
use crate::neo4j::GraphStore;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use futures::StreamExt;
use uuid::Uuid;

/// Marge de la requête entière au-delà du délai de lecture du corps (lecture du
/// déclencheur, réservation, journal).
const HANDLER_MARGIN_SECS: u64 = 20;

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Réponse générique d'un refus : même corps pour toutes les raisons.
fn rejected(status: StatusCode) -> Response {
    (status, Json(serde_json::json!({ "status": "rejected" }))).into_response()
}

fn rejected_retry(status: StatusCode, after_secs: u64) -> Response {
    let mut response = rejected(status);
    if let Ok(v) = HeaderValue::from_str(&after_secs.to_string()) {
        response.headers_mut().insert(header::RETRY_AFTER, v);
    }
    response
}

/// Ce qu'on journalise d'une livraison AUTHENTIFIÉE.
struct Delivery<'a> {
    graph: &'a dyn GraphStore,
    trigger_id: Uuid,
    source: &'a str,
    delivery_id: Option<String>,
    body_bytes: u64,
}

impl Delivery<'_> {
    /// Journalise un refus postérieur à l'authentification (best effort : un échec
    /// d'écriture est tracé, pas renvoyé) et rend la réponse générique.
    async fn refuse(&self, status: StatusCode, reason: &str) -> Response {
        let event = self.event(IngressOutcome::Rejected, Some(reason));
        if let Err(e) = self.graph.record_external_event(&event).await {
            tracing::error!(trigger_id = %self.trigger_id, error = %e, "external event journal write failed");
        }
        tracing::debug!(trigger_id = %self.trigger_id, source = self.source, reason, "external event refused");
        rejected(status)
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

    /// Libère les réservations prises (la livraison n'est pas retenue).
    async fn release(&self, keys: &[String]) {
        for key in keys {
            if let Err(e) = self
                .graph
                .release_external_delivery(self.trigger_id, key)
                .await
            {
                tracing::error!(trigger_id = %self.trigger_id, error = %e, "external event: delivery release failed");
            }
        }
    }
}

enum ReadError {
    TooLarge,
    TimedOut,
    Broken,
}

/// Lit le corps en flux : rejette DÈS que le plafond est dépassé (au plus
/// `max + un fragment` en mémoire), et sous un délai global de lecture.
async fn read_body(
    body: Body,
    max: usize,
    timeout: std::time::Duration,
) -> Result<Vec<u8>, ReadError> {
    let read = async {
        let mut stream = body.into_data_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| ReadError::Broken)?;
            if buf.len() + chunk.len() > max {
                return Err(ReadError::TooLarge);
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    };
    tokio::time::timeout(timeout, read)
        .await
        .unwrap_or(Err(ReadError::TimedOut))
}

/// POST /hooks/events/{source}/{trigger_id} — reçoit une livraison externe.
pub async fn receive_external_event(
    State(state): State<OrchestratorState>,
    Path((source, trigger_id)): Path<(String, String)>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let runtime = state.ingress.clone();
    let Some(_permit) = runtime.try_enter() else {
        runtime.refused(crate::ingress::guard::UNKNOWN_TRIGGER, "busy");
        return rejected_retry(StatusCode::SERVICE_UNAVAILABLE, 1);
    };
    // Seau global, AVANT toute lecture du graphe : un flot d'identifiants (connus ou
    // non) ne charge ni Neo4j ni le coffre au-delà de ce débit.
    if !runtime.allow_global() {
        runtime.refused(
            crate::ingress::guard::UNKNOWN_TRIGGER,
            "global_rate_limited",
        );
        let after = (60 / u64::from(runtime.config.global_rate_per_minute.max(1))).max(1);
        return rejected_retry(StatusCode::TOO_MANY_REQUESTS, after);
    }
    let limit =
        runtime.config.body_read_timeout() + std::time::Duration::from_secs(HANDLER_MARGIN_SECS);
    match tokio::time::timeout(limit, handle(&state, &source, &trigger_id, &headers, body)).await {
        Ok(response) => response,
        Err(_) => {
            tracing::warn!("external event: request timed out");
            rejected(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

async fn handle(
    state: &OrchestratorState,
    source: &str,
    raw_trigger_id: &str,
    headers: &HeaderMap,
    body: Body,
) -> Response {
    let runtime = &state.ingress;
    let config = &runtime.config;
    let unknown = crate::ingress::guard::UNKNOWN_TRIGGER;
    let graph = state.orchestrator.neo4j_arc();

    // Un identifiant mal formé, absent, d'un autre type ou d'une autre source est
    // indiscernable : 404, rien en base.
    let Ok(trigger_id) = raw_trigger_id.parse::<Uuid>() else {
        runtime.refused(unknown, "unknown_trigger");
        return rejected(StatusCode::NOT_FOUND);
    };
    let trigger = match graph.get_trigger(trigger_id).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            runtime.refused(unknown, "unknown_trigger");
            return rejected(StatusCode::NOT_FOUND);
        }
        Err(e) => {
            tracing::error!(error = %e, "external event: trigger lookup failed");
            return rejected(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let rule = match Rule::of(&trigger) {
        Some(rule) if rule.source == source => rule,
        _ => {
            runtime.refused(unknown, "unknown_trigger");
            return rejected(StatusCode::NOT_FOUND);
        }
    };

    // Le corps n'est lu qu'à partir d'une signature bien formée.
    let Some(signature) =
        header_str(headers, &rule.signature_header).filter(|s| rule::signature_well_formed(s))
    else {
        runtime.refused(trigger_id, "missing_signature");
        return rejected(StatusCode::UNAUTHORIZED);
    };
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|n| n > config.max_body_bytes as u64) {
        runtime.refused(trigger_id, "body_too_large");
        return rejected(StatusCode::PAYLOAD_TOO_LARGE);
    }
    // Échec fermé : sans secret résolu dans le coffre, rien n'est accepté. Une clé
    // `secret` en clair dans la configuration est ignorée (`Rule::of`).
    let Some(secret_name) = rule.secret_name.as_deref() else {
        runtime.refused(trigger_id, "no_secret");
        return rejected(StatusCode::FORBIDDEN);
    };
    let resolver =
        crate::chat::provider::credentials::VaultCredentialResolver::new(state.vault.clone());
    let secret = match resolver.read_vault(&Rule::grant_instance(trigger_id), secret_name) {
        Ok(Some(secret)) if secret.len() >= MIN_SECRET_BYTES => secret,
        _ => {
            runtime.refused(trigger_id, "secret_unavailable");
            return rejected(StatusCode::FORBIDDEN);
        }
    };

    let bytes = match read_body(body, config.max_body_bytes, config.body_read_timeout()).await {
        Ok(bytes) => bytes,
        Err(ReadError::TooLarge) => {
            runtime.refused(trigger_id, "body_too_large");
            return rejected(StatusCode::PAYLOAD_TOO_LARGE);
        }
        Err(ReadError::TimedOut) => {
            runtime.refused(trigger_id, "body_timeout");
            return rejected(StatusCode::REQUEST_TIMEOUT);
        }
        Err(ReadError::Broken) => {
            runtime.refused(trigger_id, "body_read_error");
            return rejected(StatusCode::BAD_REQUEST);
        }
    };
    if !crate::runner::providers::webhook::validate_github_signature(
        secret.expose(),
        &bytes,
        &signature,
    ) {
        runtime.refused(trigger_id, "invalid_signature");
        return rejected(StatusCode::UNAUTHORIZED);
    }
    drop(secret);

    // Débit par déclencheur : seulement une fois le HMAC valide. Avant, un tiers qui
    // connaît l'UUID viderait le quota du vrai émetteur sans rien signer.
    if !runtime.allow(trigger_id) {
        runtime.refused(trigger_id, "rate_limited");
        let after = (60 / u64::from(config.rate_per_minute.max(1))).max(1);
        return rejected_retry(StatusCode::TOO_MANY_REQUESTS, after);
    }

    // ── Authentifiée : à partir d'ici, chaque décision est journalisée. ──
    let delivery = Delivery {
        graph: graph.as_ref(),
        trigger_id,
        source,
        delivery_id: rule::sanitize_delivery_id(
            rule.delivery_header
                .as_deref()
                .and_then(|h| header_str(headers, h))
                .as_deref(),
        ),
        body_bytes: bytes.len() as u64,
    };
    if !trigger.enabled {
        return delivery
            .refuse(StatusCode::FORBIDDEN, "trigger_disabled")
            .await;
    }

    // Le corps signé est TOUJOURS réservé : rejouer une livraison capturée avec un
    // nouvel identifiant d'en-tête (non signé) ne passe pas. L'identifiant, s'il
    // existe, est réservé en plus.
    let mut keys = vec![format!("body:{}", body_digest(&bytes))];
    if let Some(id) = &delivery.delivery_id {
        keys.push(format!("delivery:{id}"));
    }
    let mut claimed: Vec<String> = Vec::new();
    for key in &keys {
        match graph.claim_external_delivery(trigger_id, key).await {
            Ok(true) => claimed.push(key.clone()),
            Ok(false) => {
                delivery.release(&claimed).await;
                return delivery
                    .refuse(StatusCode::CONFLICT, "replayed_delivery")
                    .await;
            }
            Err(e) => {
                tracing::error!(trigger_id = %trigger_id, error = %e, "external event: delivery claim failed");
                delivery.release(&claimed).await;
                return rejected(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    }

    let event = delivery.event(IngressOutcome::Accepted, None);
    if let Err(e) = graph.record_external_event(&event).await {
        // Jamais journalisée : on libère la réservation, sinon la relivraison de
        // l'émetteur serait refusée comme rejeu et l'événement perdu.
        tracing::error!(trigger_id = %trigger_id, error = %e, "external event journal write failed");
        delivery.release(&claimed).await;
        return rejected(StatusCode::INTERNAL_SERVER_ERROR);
    }
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "status": "accepted", "event_id": event.id })),
    )
        .into_response()
}

fn body_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::ingress::{IngressConfig, IngressRuntime};
    use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::traits::*;
    use crate::runner::{Trigger, TriggerType};
    use crate::vault::grants::{GrantScope, SecretSelector};
    use axum::http::Request;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Secret factice, rangé dans un coffre éphémère : ce n'est pas un secret réel.
    const SECRET: &str = "fixture-hmac-key-not-real-0123456789";
    const VAULT_NAME: &str = "hook-key";
    const PASS: &str = "correct horse battery staple";

    fn sign_with(key: &str, body: &[u8]) -> String {
        use hmac::{digest::KeyInit, Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).unwrap();
        mac.update(body);
        format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
    }

    fn sign(body: &[u8]) -> String {
        sign_with(SECRET, body)
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
        serde_json::json!({
            "source": "generic",
            "secret_ref": format!("vault:{VAULT_NAME}"),
            "delivery_header": "x-delivery-id"
        })
    }

    fn on() -> IngressConfig {
        IngressConfig {
            enabled: true,
            ..IngressConfig::default()
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Vault {
        /// Unlocked, the secret stored and granted to `trigger:<id>`.
        Granted,
        /// Stored and granted, then locked.
        Locked,
        /// Stored, unlocked, but granted to nobody.
        NoGrant,
    }

    struct Rig {
        state: OrchestratorState,
        graph: Arc<MockGraphStore>,
        id: Uuid,
    }

    async fn rig_full(t: Trigger, ingress: IngressConfig, vault_mode: Vault, value: &str) -> Rig {
        let graph = Arc::new(MockGraphStore::new());
        let app = crate::test_helpers::mock_app_state_with_graph(graph.clone());
        let state = crate::api::handlers::tests::mock_server_state_from(app).await;
        let mut inner: ServerState = Arc::try_unwrap(state)
            .ok()
            .expect("the fresh state is not shared");
        let vault = crate::vault::VaultService::ephemeral();
        let now = Utc::now();
        vault
            .init(PASS.into(), chrono::Duration::hours(1))
            .await
            .unwrap();
        vault.put(VAULT_NAME, value, None, now).unwrap();
        if vault_mode != Vault::NoGrant {
            vault
                .grant(
                    SecretSelector::Names([VAULT_NAME.to_string()].into()),
                    GrantScope::Provider(Rule::grant_instance(t.id)),
                    chrono::Duration::hours(1),
                    None,
                    now,
                )
                .unwrap();
        }
        if vault_mode == Vault::Locked {
            vault.lock_now();
        }
        inner.vault = vault;
        inner.ingress = Arc::new(IngressRuntime::new(ingress));
        let state = Arc::new(inner);
        graph.create_trigger(&t).await.unwrap();
        Rig {
            state,
            graph,
            id: t.id,
        }
    }

    async fn rig(t: Trigger) -> Rig {
        rig_full(t, on(), Vault::Granted, SECRET).await
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

    async fn deliver_raw(
        state: &OrchestratorState,
        source: &str,
        id: &str,
        pairs: &[(&str, &str)],
        body: Body,
    ) -> Response {
        receive_external_event(
            State(state.clone()),
            Path((source.to_string(), id.to_string())),
            headers(pairs),
            body,
        )
        .await
    }

    async fn deliver(r: &Rig, pairs: &[(&str, &str)], body: Body) -> StatusCode {
        deliver_raw(&r.state, "generic", &r.id.to_string(), pairs, body)
            .await
            .status()
    }

    async fn signed(r: &Rig, body: &[u8], delivery: &str) -> StatusCode {
        let sig = sign(body);
        deliver(
            r,
            &[("x-hub-signature-256", &sig), ("x-delivery-id", delivery)],
            Body::from(body.to_vec()),
        )
        .await
    }

    async fn journal(r: &Rig) -> Vec<ExternalEvent> {
        r.graph.list_external_events(r.id, 2000).await.unwrap()
    }

    /// Nothing at all was written: no event, no claimed delivery.
    async fn nothing_written(r: &Rig) {
        assert!(
            r.graph.external_events.read().await.is_empty(),
            "no ExternalEvent node"
        );
        assert!(
            r.graph.external_deliveries.read().await.is_empty(),
            "no ExternalDelivery node"
        );
    }

    fn bad_sig() -> String {
        format!("sha256={}", "0".repeat(64))
    }

    // ── B1: the secret lives in the vault, never in the config ──────────────

    #[tokio::test]
    async fn refuses_a_trigger_without_secret_ref_and_never_reads_the_body() {
        let r = rig(trigger(serde_json::json!({ "source": "generic" }), true)).await;
        let status = deliver(&r, &[("x-hub-signature-256", &bad_sig())], unread_body()).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        nothing_written(&r).await;
        assert_eq!(r.state.ingress.refusals_in_window(r.id), 1);
    }

    #[tokio::test]
    async fn a_plain_secret_in_the_config_is_ignored_at_reception() {
        // Written straight to the store (a legacy trigger): the creation path refuses it.
        let r = rig(trigger(
            serde_json::json!({ "source": "generic", "secret": SECRET }),
            true,
        ))
        .await;
        assert_eq!(signed(&r, b"{}", "d-1").await, StatusCode::FORBIDDEN);
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn a_vault_reference_is_never_used_as_a_literal_hmac_key() {
        let literal = format!("vault:{VAULT_NAME}");
        let r = rig(trigger(
            serde_json::json!({ "source": "generic", "secret": literal }),
            true,
        ))
        .await;
        let body = b"{}";
        let sig = sign_with(&literal, body);
        let status = deliver(&r, &[("x-hub-signature-256", &sig)], Body::from(&body[..])).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "whoever knows the name cannot sign"
        );
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn a_locked_vault_or_a_missing_grant_fails_closed() {
        for mode in [Vault::Locked, Vault::NoGrant] {
            let r = rig_full(trigger(secured(), true), on(), mode, SECRET).await;
            assert_eq!(signed(&r, b"{}", "d-1").await, StatusCode::FORBIDDEN);
            nothing_written(&r).await;
        }
    }

    #[tokio::test]
    async fn a_short_secret_is_treated_as_absent() {
        let r = rig_full(trigger(secured(), true), on(), Vault::Granted, "ten-chars!").await;
        let body = b"{}";
        let sig = sign_with("ten-chars!", body);
        let status = deliver(&r, &[("x-hub-signature-256", &sig)], Body::from(&body[..])).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn a_plain_secret_is_refused_at_creation_and_never_returned() {
        let r = rig(trigger(secured(), true)).await;
        let plan = Uuid::new_v4();
        let create = |config: serde_json::Value| {
            crate::api::handlers::create_trigger(
                State(r.state.clone()),
                Path(plan),
                axum::Extension(crate::auth::jwt::Claims::anonymous()),
                Json(serde_json::json!({"trigger_type": "webhook", "config": config})),
            )
        };
        for config in [
            serde_json::json!({"source": "github", "secret": SECRET}),
            serde_json::json!({"source": "github", "secret": SECRET, "secret_ref": "vault:k"}),
            serde_json::json!({"secret": format!("vault:{VAULT_NAME}")}),
        ] {
            let result = create(config.clone()).await;
            match result {
                Err(crate::api::handlers::AppError::BadRequest(msg)) => {
                    assert!(!msg.contains(SECRET), "the refusal does not echo the value")
                }
                other => panic!("{config} must be refused, got ok={}", other.is_ok()),
            }
        }
        let Json(ok) = create(serde_json::json!({"source": "github", "secret_ref": "vault:k"}))
            .await
            .unwrap();
        assert_eq!(ok["config"]["secret_ref"], "vault:k");
        // A legacy /api/webhooks trigger may still hold a value: it is never returned.
        let Json(legacy) = create(serde_json::json!({"secret": SECRET})).await.unwrap();
        let listed = crate::api::handlers::list_triggers(State(r.state.clone()), Path(plan))
            .await
            .unwrap()
            .0;
        for text in [legacy.to_string(), listed.to_string()] {
            assert!(!text.contains(SECRET), "{text}");
        }
        let stored = r.graph.list_triggers(plan).await.unwrap();
        assert!(stored.iter().all(|t| !format!("{t:?}").contains(SECRET)));
    }

    #[tokio::test]
    async fn a_trigger_grant_names_only_its_own_reference() {
        let r = rig(trigger(secured(), true)).await;
        let graph = r.graph.as_ref();
        let instance = Rule::grant_instance(r.id);
        let names = |n: &str| SecretSelector::Names([n.to_string()].into());
        assert!(crate::api::vault_handlers::validate_provider_grant(
            graph,
            &instance,
            &names(VAULT_NAME)
        )
        .await
        .is_ok());
        assert!(crate::api::vault_handlers::validate_provider_grant(
            graph,
            &instance,
            &names("other")
        )
        .await
        .is_err());
        assert!(crate::api::vault_handlers::validate_provider_grant(
            graph,
            &Rule::grant_instance(Uuid::new_v4()),
            &names(VAULT_NAME)
        )
        .await
        .is_err());
    }

    // ── Signature ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_missing_or_malformed_signature_is_refused_before_the_body_is_read() {
        let r = rig(trigger(secured(), true)).await;
        assert_eq!(
            deliver(&r, &[], unread_body()).await,
            StatusCode::UNAUTHORIZED
        );
        for malformed in ["sha256=00", "sha1=abc", "", "sha256=zz"] {
            let pairs = [("x-hub-signature-256", malformed)];
            let pairs: &[(&str, &str)] = if malformed.is_empty() { &[] } else { &pairs };
            assert_eq!(
                deliver(&r, pairs, unread_body()).await,
                StatusCode::UNAUTHORIZED
            );
        }
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn a_wellformed_signature_with_the_wrong_key_is_refused() {
        let r = rig(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        let sig = sign_with("another-key-of-the-same-length-000", body);
        let status = deliver(&r, &[("x-hub-signature-256", &sig)], Body::from(&body[..])).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn a_valid_signature_for_another_body_is_refused() {
        let r = rig(trigger(secured(), true)).await;
        let sig = sign(br#"{"ok":true}"#);
        let status = deliver(
            &r,
            &[("x-hub-signature-256", &sig)],
            Body::from(&br#"{"ok":false}"#[..]),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        nothing_written(&r).await;
    }

    // ── B3: no write before authentication ──────────────────────────────────

    #[tokio::test]
    async fn a_thousand_invalid_signatures_add_no_node() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig {
                rate_per_minute: 100_000,
                global_rate_per_minute: 100_000,
                ..on()
            },
            Vault::Granted,
            SECRET,
        )
        .await;
        for i in 0..1000 {
            let id = format!("d-{i}");
            let status = deliver(
                &r,
                &[("x-hub-signature-256", &bad_sig()), ("x-delivery-id", &id)],
                Body::from(&b"{}"[..]),
            )
            .await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
        }
        nothing_written(&r).await;
        assert_eq!(
            r.state.ingress.refusals_in_window(r.id),
            1000,
            "counted in memory"
        );
    }

    #[tokio::test]
    async fn unknown_or_foreign_triggers_are_not_found_and_not_journaled() {
        let r = rig(trigger(secured(), true)).await;
        let id = r.id.to_string();
        let other = Uuid::new_v4().to_string();
        for (source, target) in [
            ("github", id.as_str()),
            ("generic", &other),
            ("generic", "not-a-uuid"),
        ] {
            let status = deliver_raw(&r.state, source, target, &[], unread_body())
                .await
                .status();
            assert_eq!(status, StatusCode::NOT_FOUND, "{source}/{target}");
        }
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn refusals_before_signature_do_not_store_the_delivery_id() {
        let r = rig(trigger(secured(), true)).await;
        let status = deliver(
            &r,
            &[
                ("x-hub-signature-256", &bad_sig()),
                ("x-delivery-id", "attacker-text"),
            ],
            Body::from(&b"{}"[..]),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        nothing_written(&r).await;
        // After authentication, a delivery id that is not `[A-Za-z0-9._-]{1,128}` is dropped.
        assert_eq!(
            signed(&r, b"{\"a\":1}", "<b>x</b>").await,
            StatusCode::ACCEPTED
        );
        assert_eq!(journal(&r).await[0].delivery_id, None);
    }

    // ── Accepted, replay (B2), disabled, journal failure (M2) ────────────────

    #[tokio::test]
    async fn a_valid_signature_is_accepted_and_journaled_once() {
        let r = rig(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        assert_eq!(signed(&r, body, "d-1").await, StatusCode::ACCEPTED);
        let events = journal(&r).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].outcome, IngressOutcome::Accepted);
        assert_eq!(events[0].delivery_id.as_deref(), Some("d-1"));
        assert_eq!(events[0].body_bytes, body.len() as u64);
    }

    #[tokio::test]
    async fn a_replayed_delivery_is_refused() {
        let r = rig(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        assert_eq!(signed(&r, body, "d-7").await, StatusCode::ACCEPTED);
        assert_eq!(signed(&r, body, "d-7").await, StatusCode::CONFLICT);
        let events = journal(&r).await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].reason.as_deref(), Some("replayed_delivery"));
    }

    #[tokio::test]
    async fn replay_with_a_fresh_delivery_id_is_refused() {
        let r = rig(trigger(secured(), true)).await;
        let body = br#"{"action":"opened"}"#;
        assert_eq!(signed(&r, body, "d-1").await, StatusCode::ACCEPTED);
        assert_eq!(
            signed(&r, body, "d-2-forged").await,
            StatusCode::CONFLICT,
            "the delivery header is not signed: the body digest is what is claimed"
        );
        // Without any delivery header at all, too.
        let sig = sign(body);
        let status = deliver(&r, &[("x-hub-signature-256", &sig)], Body::from(&body[..])).await;
        assert_eq!(status, StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn a_reused_delivery_id_on_a_new_body_is_refused_and_frees_the_body() {
        let r = rig(trigger(secured(), true)).await;
        assert_eq!(signed(&r, b"{\"n\":1}", "d-1").await, StatusCode::ACCEPTED);
        assert_eq!(signed(&r, b"{\"n\":2}", "d-1").await, StatusCode::CONFLICT);
        // The body claim taken before the id clashed was released.
        assert_eq!(signed(&r, b"{\"n\":2}", "d-2").await, StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn a_disabled_trigger_journals_only_authenticated_deliveries() {
        let r = rig(trigger(secured(), false)).await;
        let status = deliver(
            &r,
            &[("x-hub-signature-256", &bad_sig())],
            Body::from(&b"{}"[..]),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        nothing_written(&r).await;
        assert_eq!(signed(&r, b"{}", "d-1").await, StatusCode::FORBIDDEN);
        let events = journal(&r).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].reason.as_deref(), Some("trigger_disabled"));
        assert!(
            r.graph.external_deliveries.read().await.is_empty(),
            "nothing claimed"
        );
    }

    #[tokio::test]
    async fn a_failed_journal_does_not_turn_the_redelivery_into_a_replay() {
        let r = rig(trigger(secured(), true)).await;
        let body = br#"{"ok":true}"#;
        r.graph
            .fail_external_event_writes
            .store(true, Ordering::SeqCst);
        assert_eq!(
            signed(&r, body, "d-9").await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(
            r.graph.external_deliveries.read().await.is_empty(),
            "the claims were released"
        );
        r.graph
            .fail_external_event_writes
            .store(false, Ordering::SeqCst);
        assert_eq!(
            signed(&r, body, "d-9").await,
            StatusCode::ACCEPTED,
            "not 409"
        );
    }

    // ── B4: size, time, concurrency, rate ───────────────────────────────────

    #[tokio::test]
    async fn an_oversized_declared_body_is_refused_before_it_is_read() {
        let r = rig(trigger(secured(), true)).await;
        let declared = (r.state.ingress.config.max_body_bytes + 1).to_string();
        let status = deliver(
            &r,
            &[
                ("x-hub-signature-256", &bad_sig()),
                ("content-length", &declared),
            ],
            unread_body(),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        nothing_written(&r).await;
    }

    #[test]
    fn the_default_body_limit_is_one_mib() {
        assert_eq!(IngressConfig::default().max_body_bytes, 1024 * 1024);
    }

    #[tokio::test]
    async fn a_streamed_body_over_the_limit_is_cut_without_buffering_it_all() {
        let r = rig(trigger(secured(), true)).await;
        const CHUNK: usize = 64 * 1024;
        let pulled = Arc::new(AtomicUsize::new(0));
        let counter = pulled.clone();
        // An endless body, with no Content-Length (chunked).
        let body = Body::from_stream(futures::stream::repeat_with(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![b'a'; CHUNK]))
        }));
        let status = deliver(&r, &[("x-hub-signature-256", &bad_sig())], body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        let max_chunks = r.state.ingress.config.max_body_bytes / CHUNK + 1;
        assert!(
            pulled.load(Ordering::SeqCst) <= max_chunks,
            "stopped at the limit: {} chunks pulled",
            pulled.load(Ordering::SeqCst)
        );
        nothing_written(&r).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_body_is_cut_by_the_read_timeout() {
        let r = rig(trigger(secured(), true)).await;
        // A body that never sends its next byte (slowloris).
        let body = Body::from_stream(futures::stream::pending::<
            Result<axum::body::Bytes, std::io::Error>,
        >());
        let started = tokio::time::Instant::now();
        let status = deliver(&r, &[("x-hub-signature-256", &bad_sig())], body).await;
        assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
        assert!(started.elapsed() >= std::time::Duration::from_secs(10));
        assert!(started.elapsed() < std::time::Duration::from_secs(11));
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn the_route_refuses_past_its_concurrency_limit() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig {
                max_concurrent: 1,
                ..on()
            },
            Vault::Granted,
            SECRET,
        )
        .await;
        let held = r.state.ingress.try_enter().expect("the only place");
        let response =
            deliver_raw(&r.state, "generic", &r.id.to_string(), &[], unread_body()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        drop(held);
        assert_eq!(signed(&r, b"{}", "d-1").await, StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn the_route_refuses_past_its_rate_per_trigger() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig {
                rate_per_minute: 2,
                ..on()
            },
            Vault::Granted,
            SECRET,
        )
        .await;
        assert_eq!(signed(&r, b"{\"n\":1}", "d-1").await, StatusCode::ACCEPTED);
        assert_eq!(signed(&r, b"{\"n\":2}", "d-2").await, StatusCode::ACCEPTED);
        assert_eq!(
            signed(&r, b"{\"n\":3}", "d-3").await,
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    // ── M1: an unauthenticated caller cannot starve the route ────────────────

    #[tokio::test]
    async fn invalid_signatures_do_not_spend_the_trigger_quota() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig {
                rate_per_minute: 3,
                global_rate_per_minute: 100_000,
                ..on()
            },
            Vault::Granted,
            SECRET,
        )
        .await;
        for _ in 0..20 {
            let status = deliver(
                &r,
                &[("x-hub-signature-256", &bad_sig())],
                Body::from(&b"{}"[..]),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::UNAUTHORIZED,
                "never 429 before the HMAC"
            );
        }
        assert_eq!(
            signed(&r, b"{\"n\":1}", "d-1").await,
            StatusCode::ACCEPTED,
            "the legitimate sender still has its whole quota"
        );
    }

    #[tokio::test]
    async fn the_global_limit_refuses_without_reading_the_graph() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig {
                global_rate_per_minute: 2,
                ..on()
            },
            Vault::Granted,
            SECRET,
        )
        .await;
        let reads = || r.graph.trigger_reads.load(Ordering::SeqCst);
        for i in 0..2 {
            assert_eq!(
                signed(&r, format!("{{\"n\":{i}}}").as_bytes(), &format!("d-{i}")).await,
                StatusCode::ACCEPTED
            );
        }
        let before = reads();
        assert_eq!(before, 2);
        for _ in 0..50 {
            let response = deliver_raw(
                &r.state,
                "generic",
                &Uuid::new_v4().to_string(),
                &[],
                unread_body(),
            )
            .await;
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(response.headers().contains_key(header::RETRY_AFTER));
        }
        assert_eq!(
            reads(),
            before,
            "the graph was not read past the global limit"
        );
        assert_eq!(journal(&r).await.len(), 2, "nothing more was written");
    }

    // ── Router: flag, public route, nothing leaks ───────────────────────────

    async fn post(
        app: &axum::Router,
        uri: &str,
        pairs: &[(&str, &str)],
        body: &[u8],
    ) -> (StatusCode, Vec<u8>) {
        let mut req = Request::builder().method("POST").uri(uri);
        for (k, v) in pairs {
            req = req.header(*k, *v);
        }
        let response = app
            .clone()
            .oneshot(req.body(Body::from(body.to_vec())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, bytes.to_vec())
    }

    #[tokio::test]
    async fn the_route_is_off_by_default_and_answers_like_an_unknown_path() {
        let r = rig_full(
            trigger(secured(), true),
            IngressConfig::default(),
            Vault::Granted,
            SECRET,
        )
        .await;
        let app = crate::api::routes::create_router(r.state.clone());
        let body = br#"{"ok":true}"#;
        let sig = sign(body);
        let pairs = [("x-hub-signature-256", sig.as_str())];
        let (status, bytes) = post(
            &app,
            &format!("/hooks/events/generic/{}", r.id),
            &pairs,
            body,
        )
        .await;
        let (unknown_status, unknown_bytes) =
            post(&app, "/hooks/no-such-route/x/y", &pairs, body).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(
            (status, bytes),
            (unknown_status, unknown_bytes),
            "indistinguishable"
        );
        nothing_written(&r).await;
    }

    #[tokio::test]
    async fn the_route_is_public_and_not_behind_require_auth() {
        let r = rig(trigger(secured(), true)).await;
        let app = crate::api::routes::create_router(r.state.clone());
        let body = br#"{"ok":true}"#;
        let sig = sign(body);
        let (status, _) = post(
            &app,
            &format!("/hooks/events/generic/{}", r.id),
            &[("x-hub-signature-256", &sig), ("x-delivery-id", "d-public")],
            body,
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "no JWT was sent");
    }

    #[derive(Clone, Default)]
    struct LogSink(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogSink {
        type Writer = LogSink;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn no_secret_reaches_a_response_body_or_a_log_line() {
        let sink = LogSink::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(sink.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        tracing::callsite::rebuild_interest_cache();

        let mut bodies = Vec::new();
        for mode in [Vault::Granted, Vault::Locked] {
            tracing::callsite::rebuild_interest_cache();
            let r = rig_full(trigger(secured(), true), on(), mode, SECRET).await;
            let app = crate::api::routes::create_router(r.state.clone());
            let uri = format!("/hooks/events/generic/{}", r.id);
            let body = br#"{"ok":true}"#;
            let sig = sign(body);
            for (pairs, payload) in [
                (
                    vec![
                        ("x-hub-signature-256", sig.as_str()),
                        ("x-delivery-id", "d-1"),
                    ],
                    &body[..],
                ),
                (
                    vec![
                        ("x-hub-signature-256", sig.as_str()),
                        ("x-delivery-id", "d-1"),
                    ],
                    &body[..],
                ),
                (vec![("x-hub-signature-256", "sha256=00")], &body[..]),
                (vec![("x-hub-signature-256", sig.as_str())], &b"other"[..]),
            ] {
                bodies.push(post(&app, &uri, &pairs, payload).await);
            }
            r.graph
                .fail_external_event_writes
                .store(true, Ordering::SeqCst);
            let fresh = br#"{"n":2}"#;
            let fresh_sig = sign(fresh);
            bodies.push(post(&app, &uri, &[("x-hub-signature-256", &fresh_sig)], fresh).await);
        }
        let statuses: Vec<_> = bodies.iter().map(|(s, _)| *s).collect();
        assert!(statuses.contains(&StatusCode::ACCEPTED), "{statuses:?}");
        assert!(statuses.contains(&StatusCode::FORBIDDEN), "{statuses:?}");
        for (_, b) in &bodies {
            let text = String::from_utf8_lossy(b);
            assert!(!text.contains(SECRET) && !text.contains(PASS), "{text}");
        }
        let logs = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        // Negative control: the capture saw the handler's own lines.
        assert!(
            logs.contains("external event refused"),
            "the capture missed the handler: {logs}"
        );
        assert!(!logs.contains(SECRET), "the secret reached the logs");
        assert!(!logs.contains(PASS));
    }
}
