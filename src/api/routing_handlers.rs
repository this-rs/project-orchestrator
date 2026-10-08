//! Routing mode settings API (decision R2): `GET/PUT /api/chat/routing` and
//! `GET/PUT/DELETE /api/projects/{slug}/routing`.
//!
//! The settings decide how much of the provider and model choice the user
//! delegates to PO (`primary | mixed | full`) and how far the learnt choices
//! are trusted (`shadow | advisory | auto`). They are stored as `LlmSetting`
//! documents (scope `global` or `project:<slug>`, key `routing`) and read with
//! a default: no document means `primary` + `shadow`, the current behaviour.
//!
//! Every mutation is reserved to a human session: `auth::middleware` refuses
//! an `agent_session` token on these paths and each handler checks again with
//! `require_human`, so the rule survives a route being moved. A refused body
//! answers 400 with a typed `code` (`invalid_routing_mode`,
//! `invalid_learning_stage`, `invalid_routing_weight`, ...), never echoing the
//! value that was sent.
//!
//! The shadow report and the decision log are read here too:
//! `GET /api/chat/routing/decisions` (newest first, paginated) and
//! `GET /api/chat/routing/report`. Both are reserved to a human session (an
//! agent must not read how it is routed, nor tune itself against the report).
//! They read an `Arc<dyn RoutingArmStore>` installed once at boot with
//! [`set_routing_store`]; until the Neo4j adapter is installed they answer an
//! empty page and an empty report. The store lives in a process-wide slot, not
//! in `ServerState`, so adding it does not touch every state constructor.
//!
//! Nothing here takes a routing decision: the settings are stored and shown
//! (also summarised in `GET /api/chat/providers`), the resolver ignores them
//! until the cognitive router lands.

use std::sync::{Arc, RwLock};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::handlers::{AppError, OrchestratorState};
use super::provider_handlers::require_human;
use crate::auth::jwt::Claims;
use crate::chat::provider::cognitive::report::{build_report, RoutingDecisionView, RoutingReport};
use crate::chat::provider::cognitive::store::{DecisionFilter, RoutingArmStore};
use crate::chat::provider::cognitive::{
    self, parse_routing_settings, EffectiveRouting, RoutingError, RoutingScope, RoutingSettings,
    ROUTING_KEY,
};
use crate::chat::provider::errors::OpenFailure;
use crate::chat::provider::settings::{project_scope, GLOBAL};

/// A refused body: 400 with the stable `code` of the rule that failed.
fn routing_error(e: RoutingError) -> AppError {
    AppError::Provider(Box::new(OpenFailure {
        status: StatusCode::BAD_REQUEST.as_u16(),
        code: e.code(),
        message: e.to_string(),
        provider_id: None,
        action: None,
        retryable: false,
        retry_after_ms: None,
    }))
}

async fn effective(
    state: &OrchestratorState,
    project_slug: Option<&str>,
) -> Result<Json<EffectiveRouting>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    cognitive::load_routing(graph.as_ref(), project_slug)
        .await
        .map(|e| Json(EffectiveRouting::from(e)))
        .map_err(AppError::Internal)
}

async fn store(
    state: &OrchestratorState,
    claims: &Claims,
    scope: &str,
    body: &Value,
) -> Result<RoutingSettings, AppError> {
    require_human(state, claims)?;
    let settings = parse_routing_settings(body).map_err(routing_error)?;
    let document = serde_json::to_string(&settings).map_err(|e| AppError::Internal(e.into()))?;
    state
        .orchestrator
        .neo4j_arc()
        .put_llm_setting(scope, ROUTING_KEY, &document)
        .await
        .map_err(AppError::Internal)?;
    Ok(settings)
}

/// GET /api/chat/routing — the global settings (scope `global`, or `default`
/// when nothing is stored).
pub async fn get_routing(
    State(state): State<OrchestratorState>,
) -> Result<Json<EffectiveRouting>, AppError> {
    effective(&state, None).await
}

/// PUT /api/chat/routing — replaces the global settings (human only).
pub async fn put_routing(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<Value>,
) -> Result<Json<EffectiveRouting>, AppError> {
    let settings = store(&state, &claims, GLOBAL, &body).await?;
    Ok(Json(EffectiveRouting::new(settings, RoutingScope::Global)))
}

/// GET /api/projects/{slug}/routing — the settings that apply to the project
/// and where they come from (`project | global | default`).
pub async fn get_project_routing(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<EffectiveRouting>, AppError> {
    effective(&state, Some(&slug)).await
}

/// PUT /api/projects/{slug}/routing — sets the project's override (human only).
pub async fn put_project_routing(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<EffectiveRouting>, AppError> {
    let settings = store(&state, &claims, &project_scope(&slug), &body).await?;
    Ok(Json(EffectiveRouting::new(settings, RoutingScope::Project)))
}

/// DELETE /api/projects/{slug}/routing — removes the project's override, so the
/// global settings apply again (human only). 404 when there was none.
pub async fn delete_project_routing(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Path(slug): Path<String>,
) -> Result<StatusCode, AppError> {
    require_human(&state, &claims)?;
    let removed = state
        .orchestrator
        .neo4j_arc()
        .delete_llm_setting(&project_scope(&slug), ROUTING_KEY)
        .await
        .map_err(AppError::Internal)?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "project '{slug}' has no routing override"
        )))
    }
}

static ROUTING_STORE: RwLock<Option<Arc<dyn RoutingArmStore>>> = RwLock::new(None);

/// Installs the store the decision log and the report read (at boot).
pub fn set_routing_store(store: Option<Arc<dyn RoutingArmStore>>) {
    *ROUTING_STORE.write().unwrap_or_else(|e| e.into_inner()) = store;
}

fn routing_store() -> Option<Arc<dyn RoutingArmStore>> {
    ROUTING_STORE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;

/// Query of `GET /api/chat/routing/decisions`.
#[derive(Debug, Deserialize)]
pub struct DecisionsQuery {
    /// Only this project's decisions.
    pub project_slug: Option<String>,
    /// Page size, 1..=200, default 50.
    pub limit: Option<usize>,
    /// Page offset.
    pub offset: Option<usize>,
    /// Only decisions at or after this RFC 3339 time.
    pub since: Option<DateTime<Utc>>,
}

/// Query of `GET /api/chat/routing/report`.
#[derive(Debug, Deserialize)]
pub struct ReportQuery {
    /// Only this project's decisions.
    pub project_slug: Option<String>,
    /// Window start (RFC 3339).
    pub from: Option<DateTime<Utc>>,
    /// Window end (RFC 3339).
    pub to: Option<DateTime<Utc>>,
}

/// One page of decisions, newest first. `has_more` is true when another page
/// follows; the store does not count, so there is no `total`.
pub async fn decisions_page(
    store: Option<&dyn RoutingArmStore>,
    query: &DecisionsQuery,
) -> anyhow::Result<Value> {
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let offset = query.offset.unwrap_or(0);
    let mut found = match store {
        Some(store) => {
            store
                .decisions(&DecisionFilter {
                    project_slug: query.project_slug.clone(),
                    since: query.since,
                    session_id: None,
                    limit: Some(limit + 1),
                    offset,
                })
                .await?
        }
        None => vec![],
    };
    let has_more = found.len() > limit;
    found.truncate(limit);
    let items: Vec<RoutingDecisionView> = found.iter().map(RoutingDecisionView::from).collect();
    Ok(serde_json::json!({
        "items": items,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
    }))
}

/// GET /api/chat/routing/decisions (human only).
pub async fn list_decisions(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Query(query): Query<DecisionsQuery>,
) -> Result<Json<Value>, AppError> {
    require_human_reader(&state, &claims)?;
    let store = routing_store();
    decisions_page(store.as_deref(), &query)
        .await
        .map(Json)
        .map_err(AppError::Internal)
}

/// GET /api/chat/routing/report (human only).
pub async fn get_report(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
    Query(query): Query<ReportQuery>,
) -> Result<Json<RoutingReport>, AppError> {
    require_human_reader(&state, &claims)?;
    let report = match routing_store() {
        Some(store) => build_report(
            store.as_ref(),
            query.project_slug.as_deref(),
            query.from,
            query.to,
        )
        .await
        .map_err(AppError::Internal)?,
        None => cognitive::report::summarise(&[], &cognitive::report::NoPrices),
    };
    Ok(Json(report))
}

fn require_human_reader(state: &OrchestratorState, claims: &Claims) -> Result<(), AppError> {
    if state.auth_config.is_none() || claims.is_human() {
        Ok(())
    } else {
        Err(AppError::Forbidden(
            "the routing log can only be read by a signed-in user".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::api::handlers::{OrchestratorState, ServerState};
    use crate::api::routes::create_router;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tower::ServiceExt;
    use uuid::Uuid;

    async fn state() -> OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
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
            auth_config: Some(crate::test_helpers::test_auth_config()),
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
        })
    }

    async fn app() -> axum::Router {
        create_router(state().await)
    }

    fn req(token: &str, method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
        let b = Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", token);
        match body {
            Some(v) => b
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        }
    }

    fn human(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
        req(&test_bearer_token(), method, uri, body)
    }

    /// A bound, live agent token (restricted profile, like a third party's).
    fn agent_bearer() -> String {
        let session = Uuid::new_v4();
        let claims = crate::auth::jwt::Claims::service_account("agent");
        let binding = crate::auth::jwt::AgentSessionBinding {
            session_id: session.to_string(),
            ceiling: Some("default".into()),
            tool_profile: Some("restricted".into()),
        };
        let (token, jti) = crate::auth::jwt::generate_session_token(
            &claims,
            Some(&binding),
            &crate::test_helpers::test_auth_config().jwt_secret,
            3600,
        )
        .unwrap();
        crate::auth::agent_tokens::register(&jti, Some(&session.to_string()));
        format!("Bearer {token}")
    }

    async fn call(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
        let resp = app.clone().oneshot(request).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, body)
    }

    fn assert_default(body: &Value, scope: &str) {
        assert_eq!(body["mode"], "primary");
        assert_eq!(body["stage"], "shadow");
        assert_eq!(body["exploration_epsilon"], 0.05);
        assert_eq!(body["cost_weight"], 0.3);
        assert_eq!(body["latency_weight"], 0.1);
        assert_eq!(body["demote_after"], 20);
        assert!(body.get("primary").is_none(), "{body}");
        assert_eq!(body["scope"], scope);
    }

    #[tokio::test]
    async fn routing_get_without_a_setting_is_the_default() {
        let app = app().await;
        let (status, body) = call(&app, human("GET", "/api/chat/routing", None)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_default(&body, "default");
        let (status, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_default(&body, "default");
    }

    #[tokio::test]
    async fn routing_put_valid_is_stored_and_read_back() {
        let app = app().await;
        let want = json!({"mode": "mixed", "stage": "advisory", "exploration_epsilon": 0.1,
                          "primary": {"provider": "claude-code", "alias": "deep"}});
        let (status, body) = call(&app, human("PUT", "/api/chat/routing", Some(want))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["mode"], "mixed");
        assert_eq!(body["stage"], "advisory");
        assert_eq!(body["exploration_epsilon"], 0.1);
        assert_eq!(body["cost_weight"], 0.3, "absent knob = default");
        assert_eq!(body["primary"]["alias"], "deep");
        assert_eq!(body["scope"], "global");
        let (_, got) = call(&app, human("GET", "/api/chat/routing", None)).await;
        assert_eq!(got, body, "GET returns exactly what PUT stored");
        // A PUT replaces the document: a knob left out goes back to its default.
        let (status, body) = call(
            &app,
            human("PUT", "/api/chat/routing", Some(json!({"mode": "full"}))),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["mode"], "full");
        assert_eq!(body["stage"], "shadow");
        assert_eq!(body["exploration_epsilon"], 0.05);
        assert!(body.get("primary").is_none());
    }

    #[tokio::test]
    async fn routing_put_invalid_is_a_typed_400_and_stores_nothing() {
        let app = app().await;
        for (body, code) in [
            (json!({"mode": "turbo"}), "invalid_routing_mode"),
            (json!({"mode": "Primary"}), "invalid_routing_mode"),
            (json!({"stage": "yolo"}), "invalid_learning_stage"),
            (
                json!({"exploration_epsilon": 0.5}),
                "invalid_routing_weight",
            ),
            (json!({"cost_weight": 1.5}), "invalid_routing_weight"),
            (json!({"latency_weight": -0.1}), "invalid_routing_weight"),
            (json!({"demote_after": 1}), "invalid_routing_weight"),
            (
                json!({"primary": {"provider": "x", "model": "m", "alias": "a"}}),
                "invalid_routing_primary",
            ),
            (json!({"cost_weight": "a lot"}), "invalid_routing_settings"),
            (json!([]), "invalid_routing_settings"),
        ] {
            for uri in ["/api/chat/routing", "/api/projects/p/routing"] {
                let (status, got) = call(&app, human("PUT", uri, Some(body.clone()))).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {body} -> {got}");
                assert_eq!(got["code"], code, "{uri} {body} -> {got}");
                assert!(got["error"].is_string());
                assert!(
                    !got.to_string().contains("turbo") && !got.to_string().contains("yolo"),
                    "a refused value is never echoed: {got}"
                );
            }
        }
        let (_, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_default(&body, "default");
    }

    #[tokio::test]
    async fn routing_mutations_are_refused_to_an_agent_token() {
        let app = app().await;
        let agent = agent_bearer();
        let body = json!({"mode": "full", "stage": "auto"});
        for (method, uri) in [
            ("PUT", "/api/chat/routing"),
            ("PUT", "/api/projects/p/routing"),
            ("DELETE", "/api/projects/p/routing"),
        ] {
            let payload = (method == "PUT").then(|| body.clone());
            let (status, got) = call(&app, req(&agent, method, uri, payload)).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri} -> {got}");
        }
        // Nothing was stored: a person still sees the default.
        let (_, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_default(&body, "default");
    }

    #[tokio::test]
    async fn routing_project_override_wins_and_delete_returns_to_global() {
        let app = app().await;
        let (status, _) = call(
            &app,
            human(
                "PUT",
                "/api/chat/routing",
                Some(json!({"mode": "mixed", "stage": "advisory"})),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Without an override the project inherits the global document.
        let (_, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_eq!(body["mode"], "mixed");
        assert_eq!(body["scope"], "global");
        // The override wins for that project only.
        let (status, body) = call(
            &app,
            human(
                "PUT",
                "/api/projects/p/routing",
                Some(json!({"mode": "full", "stage": "shadow", "cost_weight": 0.6})),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["scope"], "project");
        let (_, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_eq!(body["mode"], "full");
        assert_eq!(body["cost_weight"], 0.6);
        assert_eq!(body["scope"], "project");
        let (_, other) = call(&app, human("GET", "/api/projects/q/routing", None)).await;
        assert_eq!(other["mode"], "mixed");
        assert_eq!(other["scope"], "global");
        let (_, global) = call(&app, human("GET", "/api/chat/routing", None)).await;
        assert_eq!(global["mode"], "mixed");
        assert_eq!(global["scope"], "global");
        // DELETE returns the project to the global document.
        let (status, _) = call(&app, human("DELETE", "/api/projects/p/routing", None)).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (_, body) = call(&app, human("GET", "/api/projects/p/routing", None)).await;
        assert_eq!(body["mode"], "mixed");
        assert_eq!(body["scope"], "global");
        let (status, _) = call(&app, human("DELETE", "/api/projects/p/routing", None)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "nothing left to remove");
    }

    #[tokio::test]
    async fn routing_listing_carries_the_effective_mode_stage_and_scope() {
        let app = app().await;
        let (status, body) = call(&app, human("GET", "/api/chat/providers", None)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["routing"],
            json!({"mode": "primary", "stage": "shadow", "scope": "default"})
        );
        call(
            &app,
            human(
                "PUT",
                "/api/projects/p/routing",
                Some(json!({"mode": "full", "stage": "advisory"})),
            ),
        )
        .await;
        let (_, body) = call(
            &app,
            human("GET", "/api/chat/providers?project_slug=p", None),
        )
        .await;
        assert_eq!(
            body["routing"],
            json!({"mode": "full", "stage": "advisory", "scope": "project"})
        );
        let (_, body) = call(&app, human("GET", "/api/chat/providers", None)).await;
        assert_eq!(body["routing"]["scope"], "default", "no project asked");
    }

    #[tokio::test]
    async fn the_log_and_the_report_are_empty_without_a_store_and_human_only() {
        let app = app().await;
        let (status, body) = call(&app, human("GET", "/api/chat/routing/decisions", None)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["items"], json!([]));
        assert_eq!(body["has_more"], false);
        let (status, body) = call(&app, human("GET", "/api/chat/routing/report", None)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["decisions"], 0);
        assert!(body["agreement_rate"].is_null());
        assert!(body["estimated_cost_delta_usd"].is_null());
        assert_eq!(body["by_class"], json!([]));
        let (status, _) = call(
            &app,
            human(
                "GET",
                "/api/chat/routing/report?from=2026-10-01T00:00:00Z&to=2026-10-07T00:00:00Z",
                None,
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = call(
            &app,
            human("GET", "/api/chat/routing/report?from=nope", None),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let agent = agent_bearer();
        for uri in ["/api/chat/routing/decisions", "/api/chat/routing/report"] {
            let (status, _) = call(&app, req(&agent, "GET", uri, None)).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{uri}");
        }
    }

    #[tokio::test]
    async fn decisions_come_newest_first_in_pages_filtered_by_project() {
        use crate::chat::provider::cognitive::decision::{CognitiveDecision, Pick};
        use crate::chat::provider::cognitive::mode::{LearningStage, ProviderRoutingMode};
        use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
        use crate::chat::provider::cognitive::store::{InMemoryRoutingStore, RoutingArmStore};

        let store = InMemoryRoutingStore::new();
        for (i, project) in ["p", "p", "p", "q"].into_iter().enumerate() {
            let d = CognitiveDecision {
                id: Uuid::new_v4(),
                at: chrono::Utc::now() - chrono::Duration::seconds(100 - i as i64),
                signature: TaskSignature::utility(TaskClass::Simple, 9_000, Some(project)),
                chosen: Some(Pick::new("prov", "model")),
                score: Some(0.5),
                explored: false,
                reason: format!("d{i}"),
                alternatives: vec![],
                applied: false,
                mode: ProviderRoutingMode::Mixed,
                stage: LearningStage::Shadow,
                session_id: None,
                task_id: None,
                run_id: None,
                turn_index: None,
                outcome: None,
                used: None,
            };
            store.put_decision(&d).await.unwrap();
        }
        let query = |project: &str, limit, offset| super::DecisionsQuery {
            project_slug: Some(project.into()),
            limit: Some(limit),
            offset: Some(offset),
            since: None,
        };
        let page = super::decisions_page(Some(&store), &query("p", 2, 0))
            .await
            .unwrap();
        assert_eq!(page["items"][0]["reason"], "d2", "newest first");
        assert_eq!(page["items"][1]["reason"], "d1");
        assert_eq!(page["has_more"], true);
        let page = super::decisions_page(Some(&store), &query("p", 2, 2))
            .await
            .unwrap();
        assert_eq!(page["items"].as_array().unwrap().len(), 1);
        assert_eq!(page["items"][0]["reason"], "d0");
        assert_eq!(page["has_more"], false);
        let page = super::decisions_page(Some(&store), &query("p", 10_000, 0))
            .await
            .unwrap();
        assert_eq!(page["limit"], 200, "the page size is bounded");
    }
}
