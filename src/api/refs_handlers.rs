//! `GET /api/refs/search`: suggestions for the `#` picker of the chat composer.
//!
//! Additive and read-only. All the logic lives in [`crate::refs`]; this is the
//! glue: who is asking, which store, and how a refusal is spelled.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};

use super::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::Claims;
use crate::refs::access::{AccessPolicy, Principal};
use crate::refs::search::{search, RefSearchParams};
use crate::refs::wire::{KindsResponse, RefSearchResponse, RefsErrorBody};

/// A refused search, in the shape `tests/fixtures/refs/errors.json` pins.
pub enum RefsHttpError {
    Invalid(RefsErrorBody),
    App(AppError),
}

impl IntoResponse for RefsHttpError {
    fn into_response(self) -> Response {
        match self {
            RefsHttpError::Invalid(body) => (StatusCode::BAD_REQUEST, Json(body)).into_response(),
            RefsHttpError::App(e) => e.into_response(),
        }
    }
}

/// GET /api/refs/kinds: the kinds this server resolves (see [`KindsResponse`]).
pub async fn ref_kinds() -> Json<KindsResponse> {
    Json(KindsResponse::active())
}

/// GET /api/refs/search?q=&kinds=&project_id=&workspace_slug=&limit=
pub async fn search_refs(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Query(params): Query<RefSearchParams>,
) -> Result<Json<RefSearchResponse>, RefsHttpError> {
    let query = params
        .into_query()
        .map_err(|e| RefsHttpError::Invalid(e.into()))?;
    let principal = Principal::from_subject(&claims.sub);
    let response = search(
        state.orchestrator.neo4j_arc(),
        &AccessPolicy::open_instance(),
        &principal,
        &query,
    )
    .await
    .map_err(|e| RefsHttpError::App(e.into()))?;
    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::refs::test_support::world;
    use crate::test_helpers::{mock_app_state_with_graph, test_auth_config, test_bearer_token};
    use axum::{body::Body, http::Request};
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn app(graph: Arc<crate::neo4j::mock::MockGraphStore>) -> axum::Router {
        let app_state = mock_app_state_with_graph(graph);
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
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
        });
        create_router(state)
    }

    async fn get(router: axum::Router, uri: &str, authed: bool) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().uri(uri);
        if authed {
            req = req.header("authorization", test_bearer_token());
        }
        let resp = router
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn the_route_is_not_public() {
        let w = world().await;
        let (status, _) = get(app(w.graph.clone()).await, "/api/refs/search", false).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn the_kinds_route_is_not_public_and_tells_what_the_server_resolves() {
        let w = world().await;
        let (status, _) = get(app(w.graph.clone()).await, "/api/refs/kinds", false).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, body) = get(app(w.graph.clone()).await, "/api/refs/kinds", true).await;
        assert_eq!(status, StatusCode::OK);
        let raw = include_str!("../../tests/fixtures/refs/kinds_response.json");
        let fixture: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(body, fixture["response"]);
    }

    #[tokio::test]
    async fn a_signed_in_caller_gets_the_items_body() {
        let w = world().await;
        let (status, body) = get(
            app(w.graph.clone()).await,
            "/api/refs/search?q=refs&kinds=plan,rfc&limit=5",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let keys: Vec<_> = body.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["items"]);
        // ranked by relevance, then recency: both match "refs" as a word, so
        // the order between the two kinds is not what this test is about.
        let mut kinds: Vec<_> = body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["kind"].as_str().unwrap())
            .collect();
        kinds.sort();
        assert_eq!(kinds, ["plan", "rfc"]);
    }

    #[tokio::test]
    async fn the_scope_filters_travel_in_the_query_string() {
        let w = world().await;
        let uri = format!("/api/refs/search?kinds=plan&project_id={}", w.b);
        let (status, body) = get(app(w.graph.clone()).await, &uri, true).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["items"].as_array().unwrap().len(), 1);
        assert_eq!(body["items"][0]["id"], w.plan_b.id.to_string());
        let (_, body) = get(
            app(w.graph.clone()).await,
            "/api/refs/search?kinds=plan&workspace_slug=po",
            true,
        )
        .await;
        assert_eq!(body["items"].as_array().unwrap().len(), 1);
        assert_eq!(body["items"][0]["id"], w.plan_a.id.to_string());
    }

    #[tokio::test]
    async fn a_refused_query_is_the_400_body_of_the_error_fixture() {
        let raw = include_str!("../../tests/fixtures/refs/errors.json");
        let fixture: serde_json::Value = serde_json::from_str(raw).unwrap();
        let case = |name: &str| {
            fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == name)
                .unwrap()["body"]
                .clone()
        };
        let w = world().await;
        let long = format!("/api/refs/search?q={}", "x".repeat(201));
        let cases = [
            (long.as_str(), "search text too long"),
            ("/api/refs/search?limit=51", "search limit out of range"),
            ("/api/refs/search?limit=zero", "search limit out of range"),
        ];
        for (uri, name) in cases {
            let (status, body) = get(app(w.graph.clone()).await, uri, true).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
            assert_eq!(body, case(name), "{uri}");
        }
        let (status, body) = get(
            app(w.graph.clone()).await,
            "/api/refs/search?kinds=plan,task,step",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, case("unknown kind at index 2"));
    }

    #[tokio::test]
    async fn a_store_failure_is_a_generic_500_that_leaks_nothing() {
        let w = world().await;
        w.graph
            .fail_reads
            .lock()
            .unwrap()
            .insert("list_plans_filtered");
        let (status, body) = get(
            app(w.graph.clone()).await,
            "/api/refs/search?kinds=plan",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!body.to_string().contains("injected"));
    }
}
