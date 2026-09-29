//! Generic graph endpoints that are not scoped to a project or workspace.
//!
//! `GET /api/graph/neighborhood` — ego-graph around any entity. The query
//! logic lives in `crate::graph::neighborhood` (selection, pure) and
//! `GraphStore::get_entity_neighborhood` (bounded fetch).

use super::handlers::{AppError, OrchestratorState};
use crate::graph::neighborhood::{
    entity_kind, layer_counts, select_neighborhood, Layer, NeighborhoodParams, NeighborhoodResponse,
};
use crate::neo4j::GraphStore;
use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;

/// Entity types accepted as a centre (API names).
pub const CENTER_TYPES: &[&str] = &[
    "note",
    "decision",
    "task",
    "plan",
    "milestone",
    "project",
    "file",
    "function",
    "struct",
    "skill",
    "persona",
    "protocol",
    "feature_graph",
    "commit",
    "chat_session",
    "document",
    // accepted too, beyond the documented contract
    "trait",
    "enum",
    "step",
    "constraint",
    "release",
    "workspace",
    "workspace_milestone",
    "protocol_state",
];

/// Query string of `GET /api/graph/neighborhood`.
#[derive(Debug, Deserialize)]
pub struct NeighborhoodQuery {
    pub entity_type: String,
    pub entity_id: String,
    pub depth: Option<u32>,
    pub min_weight: Option<f64>,
    pub limit: Option<usize>,
    pub layers: Option<String>,
}

/// Validate the query, fetch the bounded candidate set and select the view.
/// Shared by the REST handler and the MCP tool.
pub async fn compute_neighborhood(
    store: &dyn GraphStore,
    q: &NeighborhoodQuery,
) -> Result<NeighborhoodResponse, AppError> {
    let entity_type = q.entity_type.trim().to_lowercase();
    if !CENTER_TYPES.contains(&entity_type.as_str()) || entity_kind(&entity_type).is_none() {
        return Err(AppError::BadRequest(format!(
            "Unsupported entity_type '{}'. Supported: {}",
            q.entity_type,
            CENTER_TYPES.join(", ")
        )));
    }
    // Matched verbatim (no trimming, no slug resolution) so `center.id`
    // echoes exactly what the client sent.
    let entity_id = q.entity_id.as_str();
    if entity_id.trim().is_empty() {
        return Err(AppError::BadRequest("entity_id is required".to_string()));
    }
    let layers = Layer::parse_csv(q.layers.as_deref()).map_err(AppError::BadRequest)?;
    let params = NeighborhoodParams::clamped(q.depth, q.min_weight, q.limit, layers);
    let not_found = || AppError::NotFound(format!("{} '{}' not found", entity_type, entity_id));

    // `stats.by_layer` counts the neighbourhood before the layers filter:
    // with a layer subset, walk all layers too (concurrently). The subset
    // walk still drives the graph itself, so depth stays "over the selected
    // layers only" and other layers never eat the fan-out budget.
    let all_params = NeighborhoodParams {
        layers: Layer::ALL.to_vec(),
        ..params.clone()
    };
    let (raw, raw_all) = if params.layers.len() == Layer::ALL.len() {
        let raw = store
            .get_entity_neighborhood(&entity_type, entity_id, &params)
            .await?;
        (raw, None)
    } else {
        let (a, b) = tokio::join!(
            store.get_entity_neighborhood(&entity_type, entity_id, &params),
            store.get_entity_neighborhood(&entity_type, entity_id, &all_params),
        );
        (a?, b?)
    };
    let raw = raw.ok_or_else(not_found)?;

    let mut resp = select_neighborhood(&raw, &params).ok_or_else(not_found)?;
    if let Some(all) = raw_all {
        resp.stats.by_layer = layer_counts(&all, params.depth, params.min_weight);
    }
    resp.center.id = q.entity_id.clone();
    Ok(resp)
}

/// GET /api/graph/neighborhood?entity_type=&entity_id=&depth=&min_weight=&limit=&layers=
pub async fn get_neighborhood(
    State(state): State<OrchestratorState>,
    Query(q): Query<NeighborhoodQuery>,
) -> Result<Json<NeighborhoodResponse>, AppError> {
    let started = std::time::Instant::now();
    let resp = compute_neighborhood(state.orchestrator.neo4j(), &q).await?;
    tracing::debug!(
        entity_type = %q.entity_type,
        nodes = resp.nodes.len(),
        edges = resp.edges.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "graph neighborhood"
    );
    Ok(Json(resp))
}

#[cfg(test)]
mod tests {
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::EventBus;
    use crate::graph::neighborhood::{InMemoryGraph, RawNode};
    use crate::neo4j::mock::MockGraphStore;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state_with_graph, test_bearer_token};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn node(id: &str, t: &str, w: f64) -> RawNode {
        RawNode {
            id: id.into(),
            node_type: t.into(),
            label: format!("label {id}"),
            subtitle: None,
            weight: w,
        }
    }

    /// note n0 with 5 synapses of decreasing weight, one of which (n1) links
    /// to a file, plus a weak synapse (0.1) to n5.
    fn sample() -> InMemoryGraph {
        let mut g = InMemoryGraph::default();
        g.add_node(node("n0", "note", 0.5));
        for i in 1..=5 {
            g.add_node(node(&format!("n{i}"), "note", 0.5));
        }
        g.add_node(node("src/lib.rs", "file", 0.9));
        g.add_edge("n0", "n1", "SYNAPSE", 0.9);
        g.add_edge("n0", "n2", "SYNAPSE", 0.8);
        g.add_edge("n0", "n3", "SYNAPSE", 0.7);
        g.add_edge("n0", "n4", "SYNAPSE", 0.6);
        g.add_edge("n0", "n5", "SYNAPSE", 0.1);
        g.add_edge("n1", "src/lib.rs", "LINKED_TO", 1.0);
        g
    }

    async fn app_with(graph: InMemoryGraph) -> axum::Router {
        let store = MockGraphStore::new();
        *store.neighborhood_graph.write().await = graph;
        let app_state = mock_app_state_with_graph(Arc::new(store));
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
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
        });
        create_router(state)
    }

    async fn get(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    fn ids(json: &serde_json::Value) -> Vec<String> {
        json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn requires_auth() {
        let app = app_with(sample()).await;
        let req = Request::builder()
            .uri("/api/graph/neighborhood?entity_type=note&entity_id=n0")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unsupported_type_is_400() {
        let app = app_with(sample()).await;
        let (status, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=chat_event&entity_id=x",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let msg = json.to_string();
        assert!(
            msg.contains("Unsupported entity_type 'chat_event'"),
            "{msg}"
        );
        assert!(msg.contains("feature_graph"), "{msg}");
    }

    #[tokio::test]
    async fn unknown_layer_is_400() {
        let app = app_with(sample()).await;
        let (status, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&layers=code,fabric",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(json.to_string().contains("Unknown layer 'fabric'"));
    }

    #[tokio::test]
    async fn unknown_id_is_404() {
        let app = app_with(sample()).await;
        let (status, _) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=does-not-exist",
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn contract_shape_and_defaults() {
        let app = app_with(sample()).await;
        let (status, json) =
            get(app, "/api/graph/neighborhood?entity_type=note&entity_id=n0").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json["center"],
            serde_json::json!({"id": "n0", "type": "note"})
        );
        let first = &json["nodes"][0];
        assert_eq!(first["id"], "n0");
        assert_eq!(first["depth"], 0);
        assert_eq!(first["layer"], "knowledge");
        assert!(first["label"].is_string());
        assert!(first["weight"].is_number());
        let edge = &json["edges"][0];
        for key in ["source", "target", "rel", "weight", "layer"] {
            assert!(!edge[key].is_null(), "edge.{key} missing");
        }
        assert_eq!(json["truncated"], false);
        assert_eq!(json["stats"]["by_type"]["note"], 6);
        assert_eq!(json["stats"]["by_type"]["file"], 1);
        assert_eq!(json["stats"]["by_rel"]["SYNAPSE"], 5);
        assert_eq!(json["stats"]["total_before_limit"], 7);
        assert_eq!(json["stats"]["by_layer"]["neural"], 5);
        assert_eq!(json["stats"]["by_layer"]["knowledge"], 1);
        assert_eq!(json["params"]["depth"], 2);
        assert_eq!(json["params"]["limit"], 120);
        // depth 2 reaches the file through n1
        let file = json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["id"] == "src/lib.rs")
            .unwrap();
        assert_eq!(file["depth"], 2);
        assert_eq!(file["layer"], "knowledge");
    }

    #[tokio::test]
    async fn depth_and_limit_are_clamped() {
        let app = app_with(sample()).await;
        let (status, json) = get(
            app.clone(),
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&depth=0&limit=0",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["params"]["depth"], 1);
        assert_eq!(json["params"]["limit"], 1);
        assert_eq!(ids(&json), vec!["n0"]);
        assert_eq!(json["truncated"], true);

        let (_, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&depth=99&limit=100000",
        )
        .await;
        assert_eq!(json["params"]["depth"], 3);
        assert_eq!(json["params"]["limit"], 400);
    }

    #[tokio::test]
    async fn by_layer_is_before_layers_filter_and_center_echoes_id() {
        let app = app_with(sample()).await;
        let (status, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=file&entity_id=src%2Flib.rs&layers=neural",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["center"]["id"], "src/lib.rs");
        assert_eq!(json["center"]["type"], "file");
        // Only neural is selected and the file has no neural edge…
        assert_eq!(ids(&json), vec!["src/lib.rs"]);
        assert_eq!(json["stats"]["total_before_limit"], 1);
        // …but by_layer still reports what the other layers hold.
        // lib.rs -LINKED_TO- n1 -SYNAPSE- n0 (depth 2)
        assert_eq!(json["stats"]["by_layer"]["knowledge"], 1);
        assert_eq!(json["stats"]["by_layer"]["neural"], 1);
        assert_eq!(json["stats"]["by_layer"]["code"], 0);
        assert_eq!(json["stats"]["by_layer"]["planning"], 0);
        assert_eq!(json["stats"]["by_layer"]["behavioral"], 0);
    }

    #[tokio::test]
    async fn min_weight_prunes() {
        let app = app_with(sample()).await;
        let (_, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&min_weight=0.65",
        )
        .await;
        let got = ids(&json);
        assert!(got.contains(&"n3".to_string()));
        assert!(!got.contains(&"n4".to_string()), "0.6 < 0.65");
        assert!(!got.contains(&"n5".to_string()));
        assert!(json["edges"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["weight"].as_f64().unwrap() >= 0.65));
    }

    #[tokio::test]
    async fn truncation_keeps_strongest() {
        let app = app_with(sample()).await;
        let (_, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&limit=3",
        )
        .await;
        assert_eq!(json["truncated"], true);
        assert_eq!(json["stats"]["total_before_limit"], 7);
        assert_eq!(ids(&json), vec!["n0", "n1", "n2"]);
    }

    #[tokio::test]
    async fn layers_filter_depth() {
        let app = app_with(sample()).await;
        let (_, json) = get(
            app,
            "/api/graph/neighborhood?entity_type=note&entity_id=n0&layers=neural&depth=3",
        )
        .await;
        assert!(!ids(&json).contains(&"src/lib.rs".to_string()));
        assert!(json["edges"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["layer"] == "neural"));
    }
}
