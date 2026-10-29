//! The quota of a provider account (QUOTA): until when it refuses, how much of its windows is used.
//!
//! * `GET    /api/chat/provider-quotas` — every instance with something known.
//! * `GET    /api/chat/providers/{id}/quota` — one instance (`quota: null` when nothing is known).
//! * `PUT    /api/chat/providers/{id}/quota` — a person's entry, for a provider that exposes
//!   nothing: `{exhausted_until_ms?, utilization?, resets_at_ms?, window?}`. HUMAN ONLY: the
//!   quota decides which provider a router may use, and an agent that could write it would choose
//!   the engine it runs on (`auth::middleware` refuses an agent token under `/api/chat/providers`
//!   and the handler checks again). The next word of the provider replaces it.
//! * `DELETE /api/chat/providers/{id}/quota` — forgets what is known (human only).
//!
//! Times are epoch milliseconds, the interface shows them in the person's time zone.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Extension, Json,
};
use serde::Serialize;
use serde_json::{json, Value};

use super::handlers::{AppError, OrchestratorState};
use super::provider_handlers::require_human;
use crate::auth::jwt::Claims;
use crate::chat::provider::quota::{ManualQuota, QuotaRegistry, QuotaView};
use crate::chat::provider::resolver::CLAUDE_CODE;
use crate::chat::provider::settings::{valid_id, GLOBAL, INSTANCE_PREFIX};
use std::sync::Arc;

/// A manual date further than this from now is a typo, not a quota.
const MAX_HORIZON_MS: u64 = 400 * 24 * 3_600_000;
/// Longest name of a window.
const MAX_WINDOW_NAME: usize = 32;

#[derive(Serialize)]
struct QuotaBody {
    /// The instant the server read the clock, so the interface does not trust its own.
    now_ms: u64,
    quota: Option<QuotaView>,
}

fn registry(state: &OrchestratorState) -> Result<Arc<QuotaRegistry>, AppError> {
    state
        .chat_manager
        .as_ref()
        .and_then(|manager| manager.cognitive_routing())
        .map(|routing| Arc::clone(&routing.quota))
        .ok_or_else(|| AppError::NotImplemented("the chat router is not running".to_string()))
}

/// Whether `id` names an instance this server knows (the built-in Claude Code, or a stored one).
async fn instance_exists(state: &OrchestratorState, id: &str) -> Result<bool, AppError> {
    if id == CLAUDE_CODE {
        return Ok(true);
    }
    if !valid_id(id) {
        return Ok(false);
    }
    Ok(state
        .orchestrator
        .neo4j()
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await
        .map_err(AppError::Internal)?
        .is_some())
}

/// GET /api/chat/provider-quotas
pub async fn list_quotas(State(state): State<OrchestratorState>) -> Result<Json<Value>, AppError> {
    let registry = registry(&state)?;
    Ok(Json(
        json!({ "now_ms": registry.now_ms(), "quotas": registry.views() }),
    ))
}

fn body_of(registry: &QuotaRegistry, id: &str) -> Result<Json<Value>, AppError> {
    Ok(Json(
        serde_json::to_value(QuotaBody {
            now_ms: registry.now_ms(),
            quota: registry.view(id),
        })
        .map_err(|e| AppError::Internal(e.into()))?,
    ))
}

/// GET /api/chat/providers/{id}/quota
pub async fn get_quota(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let registry = registry(&state)?;
    if !instance_exists(&state, &id).await? {
        return Err(AppError::NotFound(format!(
            "unknown provider instance '{id}'"
        )));
    }
    body_of(&registry, &id)
}

/// Validates what a person typed. Pure: the clock is an argument.
pub(crate) fn validate_manual(manual: &ManualQuota, now_ms: u64) -> Result<(), String> {
    if let Some(u) = manual.utilization {
        if !u.is_finite() || !(0.0..=1.0).contains(&u) {
            return Err("utilization is a share between 0 and 1".to_string());
        }
    }
    for (name, at) in [
        ("exhausted_until_ms", manual.exhausted_until_ms),
        ("resets_at_ms", manual.resets_at_ms),
    ] {
        if let Some(at) = at {
            if at > now_ms + MAX_HORIZON_MS {
                return Err(format!("{name} is more than 400 days away"));
            }
        }
    }
    if let Some(window) = &manual.window {
        if window.is_empty()
            || window.len() > MAX_WINDOW_NAME
            || !window
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err("window is a short name (letters, digits, - and _)".to_string());
        }
    }
    if manual.exhausted_until_ms.is_none()
        && manual.utilization.is_none()
        && manual.resets_at_ms.is_none()
    {
        return Err("say something: exhausted_until_ms, utilization or resets_at_ms".to_string());
    }
    Ok(())
}

/// PUT /api/chat/providers/{id}/quota — human only.
pub async fn put_quota(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
    Json(manual): Json<ManualQuota>,
) -> Result<Json<Value>, AppError> {
    require_human(&state, &claims)?;
    let registry = registry(&state)?;
    if !instance_exists(&state, &id).await? {
        return Err(AppError::NotFound(format!(
            "unknown provider instance '{id}'"
        )));
    }
    validate_manual(&manual, registry.now_ms()).map_err(AppError::BadRequest)?;
    registry.set_manual(&id, &manual, &claims.email);
    registry.persist(&id).await.map_err(AppError::Internal)?;
    body_of(&registry, &id)
}

/// DELETE /api/chat/providers/{id}/quota — human only.
pub async fn delete_quota(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    let registry = registry(&state)?;
    registry.clear(&id);
    registry.persist(&id).await.map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::auth::jwt::{generate_session_token, Claims};
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_auth_config, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode as HttpStatus};
    use tower::ServiceExt;

    const NOW: u64 = 1_792_500_000_000;

    fn manual(until: Option<u64>) -> ManualQuota {
        ManualQuota {
            exhausted_until_ms: until,
            utilization: None,
            resets_at_ms: None,
            window: None,
        }
    }

    #[test]
    fn a_plausible_entry_is_accepted() {
        assert!(validate_manual(&manual(Some(NOW + 3_600_000)), NOW).is_ok());
        let gauge = ManualQuota {
            exhausted_until_ms: None,
            utilization: Some(0.4),
            resets_at_ms: Some(NOW + 86_400_000),
            window: Some("seven_day".into()),
        };
        assert!(validate_manual(&gauge, NOW).is_ok());
    }

    #[test]
    fn a_typo_is_refused_with_its_reason() {
        assert!(validate_manual(&manual(None), NOW).is_err(), "nothing said");
        assert!(validate_manual(&manual(Some(NOW + 500 * 86_400_000)), NOW)
            .unwrap_err()
            .contains("400 days"));
        let mut bad = manual(Some(NOW + 1000));
        bad.utilization = Some(1.5);
        assert!(validate_manual(&bad, NOW).is_err());
        let mut bad = manual(Some(NOW + 1000));
        bad.window = Some("a b".into());
        assert!(validate_manual(&bad, NOW).is_err());
    }

    /// A token an agent holds: the same person, but minted for a session.
    fn agent_bearer() -> String {
        let person = Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "test@ffs.holdings".into(),
            name: "Test User".into(),
            iat: 0,
            exp: i64::MAX,
            token_type: None,
            scope: None,
            jti: None,
        };
        let (token, jti) =
            generate_session_token(&person, None, "test-secret-key-minimum-32-chars!!", 28_800)
                .expect("an agent session token");
        // The mint path registers the `jti`; without it the token is merely revoked (401) and the
        // test would never reach the check it is about.
        crate::auth::agent_tokens::register(&jti, None);
        format!("Bearer {token}")
    }

    /// A router whose chat side is NOT running: enough to prove WHO is refused, because the
    /// refusal must come before anything the router owns is even looked up.
    async fn app_without_chat() -> axum::Router {
        create_router(state_without_chat().await)
    }

    /// The same state, unwrapped: a handler called DIRECTLY, with no middleware in front of it.
    async fn state_without_chat() -> OrchestratorState {
        let orchestrator = Arc::new(Orchestrator::new(mock_app_state()).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        })
    }

    fn put(bearer: &str) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri("/api/chat/providers/claude-code/quota")
            .header("authorization", bearer)
            .header("content-type", "application/json")
            .body(Body::from(r#"{"exhausted_until_ms":1792600000000}"#))
            .unwrap()
    }

    /// QUOTA: the quota decides which provider a router may use, so an AGENT must not write it —
    /// it would choose the engine it runs on. 403, and nothing of the registry is reached.
    ///
    /// The person's own token gets past that gate on the very same route: the test would pass just
    /// as well if everything were refused, so it checks both sides. With the chat router stopped,
    /// a person reaches the handler and is told so (501) — which also pins the ORDER: were
    /// `require_human` to move after the registry lookup, the agent would get 501 here too.
    #[tokio::test]
    async fn an_agent_token_may_not_write_a_quota_while_a_person_may() {
        let refused = app_without_chat()
            .await
            .oneshot(put(&agent_bearer()))
            .await
            .unwrap();
        assert_eq!(
            refused.status(),
            HttpStatus::FORBIDDEN,
            "an agent token must never write a quota"
        );

        let person = app_without_chat()
            .await
            .oneshot(put(&test_bearer_token()))
            .await
            .unwrap();
        assert_ne!(
            person.status(),
            HttpStatus::FORBIDDEN,
            "a signed-in person is not refused on the same route"
        );
        assert_eq!(
            person.status(),
            HttpStatus::NOT_IMPLEMENTED,
            "the person reached the handler, which says the chat router is not running"
        );
    }

    /// The handler's OWN refusal, with no middleware in front of it.
    ///
    /// Why this exists on top of the router test: the auth middleware already refuses an agent
    /// token under `/api/chat/providers`, so the router test passes even when `require_human` is
    /// deleted from this handler — a mutation proved it (the gate was belt alone, and the braces
    /// were untested). Calling the handler directly is what pins the second check.
    ///
    /// It also pins the ORDER: a person gets `NotImplemented` (the chat router is stopped), which
    /// is only reachable once `require_human` has let them through.
    #[tokio::test]
    async fn the_handler_refuses_an_agent_by_itself_without_the_middleware() {
        let state = state_without_chat().await;
        let person = Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "test@ffs.holdings".into(),
            name: "Test User".into(),
            iat: 0,
            exp: i64::MAX,
            token_type: None,
            scope: None,
            jti: None,
        };
        let agent = Claims {
            token_type: Some(crate::auth::jwt::TOKEN_TYPE_AGENT_SESSION.to_string()),
            ..person.clone()
        };
        let body = manual(Some(NOW + 3_600_000));

        let refused = put_quota(
            State(Arc::clone(&state)),
            Extension(agent),
            Path(CLAUDE_CODE.to_string()),
            Json(body.clone()),
        )
        .await
        .expect_err("an agent must not write a quota");
        assert!(
            matches!(refused, AppError::Forbidden(_)),
            "expected Forbidden, got {refused:?}"
        );

        let person_err = put_quota(
            State(state),
            Extension(person),
            Path(CLAUDE_CODE.to_string()),
            Json(body),
        )
        .await
        .expect_err("the chat router is not running in this state");
        assert!(
            matches!(person_err, AppError::NotImplemented(_)),
            "a person is let through to the registry lookup, got {person_err:?}"
        );
    }

    /// The route is under the prefix whose MUTATIONS the middleware reserves to a person, while
    /// reads stay open. A new route mounted outside that prefix would make this red.
    #[test]
    fn the_quota_route_is_a_human_only_mutation_but_its_read_is_open() {
        use crate::auth::middleware::is_human_only_mutation;
        use axum::http::Method;
        const PATH: &str = "/api/chat/providers/claude-code/quota";
        assert!(is_human_only_mutation(&Method::PUT, PATH));
        assert!(is_human_only_mutation(&Method::DELETE, PATH));
        assert!(
            !is_human_only_mutation(&Method::GET, PATH),
            "reads stay open"
        );
        // The listing is a read, and deliberately NOT under the `/api/chat/providers` prefix.
        assert!(!is_human_only_mutation(
            &Method::GET,
            "/api/chat/provider-quotas"
        ));
    }
}
