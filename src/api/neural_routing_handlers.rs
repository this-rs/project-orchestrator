//! API handlers for Neural Routing.
//!
//! These handlers back the MCP `neural_routing` mega-tool and provide
//! the REST surface for neural route learning status and configuration.

use super::handlers::{AppError, OrchestratorState};
use axum::{extract::State, Json};
use neural_routing_runtime::config::{NeuralRoutingConfig, RoutingMode};
use neural_routing_runtime::NNMetricsSnapshot;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// Request / Response types
// ============================================================================

/// Response for GET /api/neural-routing/status
#[derive(Debug, Serialize)]
pub struct NeuralRoutingStatusResponse {
    pub enabled: bool,
    pub mode: String,
    pub cpu_guard_paused: bool,
    pub metrics: NNMetricsSnapshot,
}

/// Response for GET /api/neural-routing/config
#[derive(Debug, Serialize)]
pub struct NeuralRoutingConfigResponse {
    pub config: NeuralRoutingConfig,
}

/// Request body for PUT /api/neural-routing/mode
#[derive(Debug, Deserialize)]
pub struct SetModeRequest {
    /// "nn" or "full"
    pub mode: String,
}

/// Request body for PUT /api/neural-routing/config
#[derive(Debug, Deserialize)]
pub struct UpdateConfigRequest {
    pub enabled: Option<bool>,
    pub mode: Option<String>,
    pub inference_timeout_ms: Option<u64>,
    pub nn_fallback: Option<bool>,
    pub collection_enabled: Option<bool>,
    pub collection_buffer_size: Option<usize>,
    pub nn_top_k: Option<usize>,
    pub nn_min_similarity: Option<f32>,
    pub nn_max_route_age_days: Option<u32>,
}

/// Generic success response
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub ok: bool,
    pub message: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/neural-routing/status
///
/// Returns current neural routing status including mode, CPU guard state, and metrics.
pub async fn get_status(
    State(state): State<OrchestratorState>,
) -> Result<Json<NeuralRoutingStatusResponse>, AppError> {
    let router = state.neural_router.read().await;
    let config = router.config();
    let metrics = router.nn_router().metrics().snapshot();

    let mode_str = match config.mode {
        RoutingMode::NN => "nn",
        RoutingMode::Full => "full",
    };

    Ok(Json(NeuralRoutingStatusResponse {
        enabled: config.enabled,
        mode: mode_str.to_string(),
        cpu_guard_paused: router.cpu_guard().is_paused(),
        metrics,
    }))
}

/// GET /api/neural-routing/config
///
/// Returns the full neural routing configuration.
pub async fn get_config(
    State(state): State<OrchestratorState>,
) -> Result<Json<NeuralRoutingConfigResponse>, AppError> {
    let router = state.neural_router.read().await;
    let config = router.config().clone();

    Ok(Json(NeuralRoutingConfigResponse { config }))
}

/// POST /api/neural-routing/enable
///
/// Enable neural routing at runtime.
pub async fn enable(
    State(state): State<OrchestratorState>,
) -> Result<Json<SuccessResponse>, AppError> {
    let mut router = state.neural_router.write().await;
    let mut config = router.config().clone();
    config.enabled = true;
    router.update_config(config);

    tracing::info!("Neural routing enabled via API");

    Ok(Json(SuccessResponse {
        ok: true,
        message: "Neural routing enabled".to_string(),
    }))
}

/// POST /api/neural-routing/disable
///
/// Disable neural routing at runtime.
pub async fn disable(
    State(state): State<OrchestratorState>,
) -> Result<Json<SuccessResponse>, AppError> {
    let mut router = state.neural_router.write().await;
    let mut config = router.config().clone();
    config.enabled = false;
    router.update_config(config);

    // Also disable trajectory collection when routing is disabled
    if let Some(ref collector) = *state.trajectory_collector.read().unwrap() {
        collector.set_enabled(false);
    }

    tracing::info!("Neural routing disabled via API");

    Ok(Json(SuccessResponse {
        ok: true,
        message: "Neural routing disabled".to_string(),
    }))
}

/// PUT /api/neural-routing/mode
///
/// Set routing mode: "nn" (NN only) or "full" (Policy Net + NN fallback).
pub async fn set_mode(
    State(state): State<OrchestratorState>,
    Json(req): Json<SetModeRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let new_mode = match req.mode.as_str() {
        "nn" => RoutingMode::NN,
        "full" => RoutingMode::Full,
        other => {
            return Err(AppError::BadRequest(format!(
                "Invalid mode '{}': expected 'nn' or 'full'",
                other
            )));
        }
    };

    let mut router = state.neural_router.write().await;
    let mut config = router.config().clone();
    config.mode = new_mode.clone();
    router.update_config(config);

    let mode_str = match new_mode {
        RoutingMode::NN => "nn",
        RoutingMode::Full => "full",
    };
    tracing::info!(mode = mode_str, "Neural routing mode changed via API");

    Ok(Json(SuccessResponse {
        ok: true,
        message: format!("Neural routing mode set to '{}'", mode_str),
    }))
}

/// PUT /api/neural-routing/config
///
/// Update neural routing configuration (partial update).
pub async fn update_config(
    State(state): State<OrchestratorState>,
    Json(req): Json<UpdateConfigRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let mut router = state.neural_router.write().await;
    let mut config = router.config().clone();

    if let Some(enabled) = req.enabled {
        config.enabled = enabled;
    }
    if let Some(ref mode) = req.mode {
        config.mode = match mode.as_str() {
            "nn" => RoutingMode::NN,
            "full" => RoutingMode::Full,
            other => {
                return Err(AppError::BadRequest(format!(
                    "Invalid mode '{}': expected 'nn' or 'full'",
                    other
                )));
            }
        };
    }
    if let Some(timeout_ms) = req.inference_timeout_ms {
        config.inference.timeout_ms = timeout_ms;
    }
    if let Some(nn_fallback) = req.nn_fallback {
        config.inference.nn_fallback = nn_fallback;
    }
    if let Some(collection_enabled) = req.collection_enabled {
        config.collection.enabled = collection_enabled;
    }
    if let Some(buffer_size) = req.collection_buffer_size {
        config.collection.buffer_size = buffer_size;
    }
    if let Some(top_k) = req.nn_top_k {
        config.nn.top_k = top_k;
    }
    if let Some(min_sim) = req.nn_min_similarity {
        config.nn.min_similarity = min_sim;
    }
    if let Some(max_age) = req.nn_max_route_age_days {
        config.nn.max_route_age_days = max_age;
    }

    router.update_config(config.clone());

    // Propagate collection.enabled to the TrajectoryCollector (lazy-init if needed)
    if let Some(collection_enabled) = req.collection_enabled {
        let mut tc_guard = state.trajectory_collector.write().unwrap();
        if let Some(ref collector) = *tc_guard {
            // Collector already exists — just toggle
            collector.set_enabled(collection_enabled);
            tracing::info!(
                enabled = collection_enabled,
                "Trajectory collector toggled via config update"
            );
        } else if collection_enabled {
            // Collector doesn't exist yet — create it on demand
            if let Some(ref store) = state.trajectory_store_neo4j {
                let coll_config = neural_routing_runtime::config::CollectionConfig {
                    enabled: true,
                    buffer_size: config.collection.buffer_size,
                    stale_session_timeout_secs: config.collection.stale_session_timeout_secs,
                };
                let (collector, _handle) = neural_routing_runtime::TrajectoryCollector::new(
                    store.clone(),
                    &coll_config,
                    None,
                );
                let collector = Arc::new(collector);

                // Propagate to the ChatManager so it can record decisions
                // and finalize trajectories on session close.
                if let Some(ref cm) = state.chat_manager {
                    cm.set_trajectory_collector(collector.clone());
                    tracing::info!("Trajectory collector propagated to ChatManager");
                }

                *tc_guard = Some(collector);
                tracing::info!(
                    buffer_size = coll_config.buffer_size,
                    "Trajectory collector created at runtime via config update"
                );
            } else {
                tracing::warn!(
                    "Cannot create trajectory collector: no Neo4j trajectory store available"
                );
            }
        }
    }

    tracing::info!("Neural routing config updated via API");

    Ok(Json(SuccessResponse {
        ok: true,
        message: "Neural routing configuration updated".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, mock_neural_router};
    use std::sync::Arc;

    /// Build a minimal `OrchestratorState` for handler-level tests.
    async fn test_state() -> OrchestratorState {
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
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 0,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        })
    }

    #[tokio::test]
    async fn test_get_status() {
        let state = test_state().await;

        let result = get_status(State(state)).await;
        assert!(result.is_ok(), "get_status should succeed");

        let resp = result.unwrap().0;
        // mock_neural_router creates a router with enabled=false and default mode (NN)
        assert!(!resp.enabled);
        assert_eq!(resp.mode, "nn");
        // metrics should be zeroed on a fresh router
        assert_eq!(resp.metrics.total_queries, 0);
        assert_eq!(resp.metrics.hits, 0);
    }

    #[tokio::test]
    async fn test_enable_disable_roundtrip() {
        let state = test_state().await;

        // Initially disabled (mock default)
        let status = get_status(State(state.clone())).await.unwrap().0;
        assert!(!status.enabled);

        // Disable explicitly (should be idempotent)
        let resp = disable(State(state.clone())).await.unwrap().0;
        assert!(resp.ok);

        let status = get_status(State(state.clone())).await.unwrap().0;
        assert!(!status.enabled);

        // Enable
        let resp = enable(State(state.clone())).await.unwrap().0;
        assert!(resp.ok);
        assert!(resp.message.contains("enabled"));

        let status = get_status(State(state.clone())).await.unwrap().0;
        assert!(status.enabled);

        // Disable again
        let resp = disable(State(state.clone())).await.unwrap().0;
        assert!(resp.ok);

        let status = get_status(State(state.clone())).await.unwrap().0;
        assert!(!status.enabled);
    }

    #[tokio::test]
    async fn test_set_mode_nn() {
        let state = test_state().await;

        // Set mode to "nn"
        let req = SetModeRequest {
            mode: "nn".to_string(),
        };
        let resp = set_mode(State(state.clone()), Json(req)).await.unwrap().0;
        assert!(resp.ok);
        assert!(resp.message.contains("nn"));

        // Verify config reflects the change
        let config_resp = get_config(State(state.clone())).await.unwrap().0;
        assert_eq!(config_resp.config.mode, RoutingMode::NN);

        // Set mode to "full"
        let req = SetModeRequest {
            mode: "full".to_string(),
        };
        let resp = set_mode(State(state.clone()), Json(req)).await.unwrap().0;
        assert!(resp.ok);
        assert!(resp.message.contains("full"));

        let config_resp = get_config(State(state.clone())).await.unwrap().0;
        assert_eq!(config_resp.config.mode, RoutingMode::Full);
    }

    #[tokio::test]
    async fn test_set_mode_invalid() {
        let state = test_state().await;

        let req = SetModeRequest {
            mode: "invalid_mode".to_string(),
        };
        let result = set_mode(State(state), Json(req)).await;
        assert!(result.is_err(), "invalid mode should return an error");

        match result.unwrap_err() {
            AppError::BadRequest(msg) => {
                assert!(msg.contains("invalid_mode"));
                assert!(msg.contains("expected 'nn' or 'full'"));
            }
            other => panic!("Expected BadRequest, got {:?}", other),
        }
    }
}
