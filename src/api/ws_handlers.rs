//! WebSocket handlers for real-time event notifications
//!
//! Multiplexes two event streams into a single WebSocket connection:
//! - **CRUD events**: entity mutations (create/update/delete/link/unlink)
//! - **Graph events**: graph visualization mutations (node/edge/reinforcement/activation)
//!
//! Events from both local and remote sources (NATS) arrive via the local
//! broadcast bus thanks to the NATS→local bridge in `HybridEmitter`.
//! WS handlers only need to subscribe to the local bus.

use super::handlers::OrchestratorState;
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
use crate::events::graph::GraphEvent;
use crate::events::CrudEvent;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::HashSet;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

/// Query parameters for filtering WebSocket events
#[derive(Debug, Deserialize, Default)]
pub struct WsQuery {
    /// Comma-separated entity types to subscribe to (e.g. "plan,task,note")
    pub entity_types: Option<String>,
    /// Filter by project ID
    pub project_id: Option<String>,
    /// Comma-separated graph layers to subscribe to (e.g. "knowledge,neural,fabric")
    /// When absent, all graph events are forwarded. When present, only events
    /// matching the specified layers are sent.
    pub layers: Option<String>,
    /// One-time ticket for auth (fallback when cookies aren't sent on WS upgrade)
    pub ticket: Option<String>,
}

/// WebSocket upgrade handler for `/ws/events`
///
/// Authentication: validates `refresh_token` cookie (or ticket fallback) BEFORE upgrade.
/// - Valid cookie/ticket (or no-auth mode) → upgrade + send `auth_ok` immediately
/// - No credentials or invalid → reject with 401 (no upgrade)
pub async fn ws_events(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Query(query): Query<WsQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let entity_filter: Option<HashSet<String>> = query.entity_types.map(|types| {
        types
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let project_filter = query.project_id;
    let layer_filter: Option<HashSet<String>> = query.layers.map(|layers| {
        layers
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });

    // Pre-upgrade auth: cookie first, then ticket fallback
    let has_cookie = headers.get(axum::http::header::COOKIE).is_some();
    let has_ticket = query.ticket.is_some();
    info!(
        has_cookie = has_cookie,
        has_ticket = has_ticket,
        "WS /ws/events upgrade request received"
    );

    let neo4j = state.orchestrator.neo4j_arc();
    let auth_result = super::ws_auth::ws_authenticate(
        &headers,
        &state.auth_config,
        &neo4j,
        query.ticket.as_deref(),
        &state.ws_ticket_store,
    )
    .await;

    match auth_result {
        CookieAuthResult::Authenticated(claims) => {
            info!(email = %claims.email, "WS /ws/events: authenticated, upgrading");
            ws.on_upgrade(move |socket| {
                handle_ws_preauthed(
                    socket,
                    state,
                    entity_filter,
                    project_filter,
                    layer_filter,
                    claims,
                )
            })
            .into_response()
        }
        CookieAuthResult::Invalid(reason) => {
            warn!(reason = %reason, "WS /ws/events: auth REJECTED (401)");
            StatusCode::UNAUTHORIZED.into_response()
        }
    }
}

/// Check if the event passes entity_type and project_id filters.
fn passes_filters(
    event: &CrudEvent,
    entity_filter: &Option<HashSet<String>>,
    project_filter: &Option<String>,
) -> bool {
    // Apply entity_type filter
    if let Some(ref filter) = entity_filter {
        let entity_str = serde_json::to_value(&event.entity_type)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();
        if !filter.contains(&entity_str) {
            return false;
        }
    }

    // Apply project_id filter
    if let Some(ref pid) = project_filter {
        match &event.project_id {
            Some(event_pid) if event_pid == pid => {}
            Some(_) => return false,
            // Events without project_id pass through (global events)
            None => {}
        }
    }

    true
}

/// Check if a graph event passes layer and project_id filters.
fn passes_graph_filters(
    event: &GraphEvent,
    layer_filter: &Option<HashSet<String>>,
    project_filter: &Option<String>,
) -> bool {
    // Apply layer filter
    if let Some(ref filter) = layer_filter {
        let layer_str = serde_json::to_value(&event.layer)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();
        if !filter.contains(&layer_str) {
            return false;
        }
    }

    // Apply project_id filter
    if let Some(ref pid) = project_filter {
        if &event.project_id != pid {
            return false;
        }
    }

    true
}

/// Handle a WebSocket connection that was pre-authenticated via cookie.
///
/// Waits for a `"ready"` message from the client before sending `auth_ok`.
/// This prevents a race condition with the Tauri WebSocket plugin where
/// `auth_ok` can be lost if sent before the client's message listener is
/// registered (the plugin's `connect()` resolves before `addListener()`).
async fn handle_ws_preauthed(
    mut socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
    layer_filter: Option<HashSet<String>>,
    claims: Claims,
) {
    // Wait for client "ready" signal before sending auth_ok
    super::ws_auth::wait_ready_then_auth_ok(&mut socket, &claims).await;
    handle_ws_loop(
        socket,
        state,
        entity_filter,
        project_filter,
        layer_filter,
        claims,
    )
    .await;
}

/// Maximum graph events in a single batch before flushing immediately
const GRAPH_BATCH_MAX_SIZE: usize = 10;
/// Maximum time to wait before flushing a non-empty graph batch
const GRAPH_BATCH_WINDOW: Duration = Duration::from_millis(100);

/// Flush a batch of graph events as a single WebSocket message.
///
/// Sends `{"kind":"graph_batch","events":[...],"count":N}` to the client.
/// Returns `true` on success, `false` if the WebSocket send failed.
async fn flush_graph_batch(
    ws_sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    batch: &mut Vec<GraphEvent>,
) -> bool {
    if batch.is_empty() {
        return true;
    }

    let msg = serde_json::json!({
        "kind": "graph_batch",
        "events": batch,
        "count": batch.len(),
    });
    batch.clear();

    match serde_json::to_string(&msg) {
        Ok(json) => {
            if ws_sender.send(Message::Text(json.into())).await.is_err() {
                debug!("WebSocket send failed during graph batch flush");
                return false;
            }
        }
        Err(e) => {
            warn!("Failed to serialize graph batch: {}", e);
        }
    }

    true
}

/// Main event loop for an authenticated WebSocket connection.
///
/// Multiplexes two broadcast channels into a single WebSocket stream:
/// - **CRUD events** from `event_bus.subscribe()` — sent immediately (low volume)
/// - **Graph events** from `event_bus.subscribe_graph()` — batched to prevent flood
///
/// Graph event batching: events are collected into a buffer and flushed either:
/// - When the buffer reaches `GRAPH_BATCH_MAX_SIZE` (10) events, OR
/// - After `GRAPH_BATCH_WINDOW` (100ms) since the first buffered event
///
/// This ensures high-frequency mutations (sync pipeline, bulk reinforcement) don't
/// overwhelm the WebSocket connection while maintaining < 100ms latency.
async fn handle_ws_loop(
    socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
    layer_filter: Option<HashSet<String>>,
    claims: Claims,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();
    let mut event_rx = state.event_bus.subscribe();
    let mut graph_rx = state.event_bus.subscribe_graph();

    // Ping interval (30s)
    let mut ping_interval = interval(Duration::from_secs(30));
    // Skip the first immediate tick
    ping_interval.tick().await;

    // Graph event batch buffer
    let mut graph_batch: Vec<GraphEvent> = Vec::with_capacity(GRAPH_BATCH_MAX_SIZE);
    let batch_timer = tokio::time::sleep(GRAPH_BATCH_WINDOW);
    tokio::pin!(batch_timer);
    let mut batch_timer_active = false;

    debug!(
        email = %claims.email,
        entity_filter = ?entity_filter,
        project_filter = ?project_filter,
        layer_filter = ?layer_filter,
        "WebSocket events client authenticated"
    );

    loop {
        tokio::select! {
            // Forward CRUD events (local + NATS-bridged) immediately
            result = event_rx.recv() => {
                match result {
                    Ok(event) => {
                        if !passes_filters(&event, &entity_filter, &project_filter) {
                            continue;
                        }

                        match serde_json::to_string(&event) {
                            Ok(json) => {
                                if ws_sender.send(Message::Text(json.into())).await.is_err() {
                                    debug!("WebSocket send failed, client disconnected");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to serialize CrudEvent: {}", e);
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "WebSocket CRUD client lagged, skipping events");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        debug!("CRUD event bus closed, shutting down WebSocket");
                        break;
                    }
                }
            }

            // Collect graph events into batch buffer
            result = graph_rx.recv() => {
                match result {
                    Ok(event) => {
                        if !passes_graph_filters(&event, &layer_filter, &project_filter) {
                            continue;
                        }

                        graph_batch.push(event);

                        // Start the batch window timer on first event
                        if !batch_timer_active {
                            batch_timer.as_mut().reset(tokio::time::Instant::now() + GRAPH_BATCH_WINDOW);
                            batch_timer_active = true;
                        }

                        // Flush immediately if batch is full
                        if graph_batch.len() >= GRAPH_BATCH_MAX_SIZE {
                            if !flush_graph_batch(&mut ws_sender, &mut graph_batch).await {
                                break;
                            }
                            batch_timer_active = false;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "WebSocket graph client lagged, skipping events");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        debug!("Graph event bus closed, shutting down WebSocket");
                        break;
                    }
                }
            }

            // Flush the graph batch when the window expires
            _ = &mut batch_timer, if batch_timer_active => {
                if !flush_graph_batch(&mut ws_sender, &mut graph_batch).await {
                    break;
                }
                batch_timer_active = false;
            }

            // Send periodic pings to detect dead clients
            _ = ping_interval.tick() => {
                if ws_sender.send(Message::Ping(vec![].into())).await.is_err() {
                    debug!("Ping failed, client disconnected");
                    break;
                }
            }

            // Handle incoming messages from the client (Pong, Close)
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Pong(_))) => {
                        // Client is alive
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        debug!("WebSocket client disconnected");
                        break;
                    }
                    Some(Err(e)) => {
                        debug!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {
                        // Ignore text/binary messages from clients
                    }
                }
            }
        }
    }

    debug!("WebSocket connection closed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::graph::{GraphEventType, GraphLayer};
    use crate::events::{CrudAction, EntityType};

    // ================================================================
    // passes_filters (CRUD events)
    // ================================================================

    fn make_crud_event(entity_type: EntityType, project_id: Option<&str>) -> CrudEvent {
        let mut e = CrudEvent::new(entity_type, CrudAction::Created, "test-id");
        if let Some(pid) = project_id {
            e = e.with_project_id(pid);
        }
        e
    }

    #[test]
    fn test_passes_filters_no_filters() {
        let event = make_crud_event(EntityType::Plan, Some("proj-1"));
        assert!(passes_filters(&event, &None, &None));
    }

    #[test]
    fn test_passes_filters_entity_match() {
        let event = make_crud_event(EntityType::Plan, None);
        let filter: Option<HashSet<String>> = Some(
            ["plan".to_string(), "task".to_string()]
                .into_iter()
                .collect(),
        );
        assert!(passes_filters(&event, &filter, &None));
    }

    #[test]
    fn test_passes_filters_entity_no_match() {
        let event = make_crud_event(EntityType::Note, None);
        let filter: Option<HashSet<String>> = Some(
            ["plan".to_string(), "task".to_string()]
                .into_iter()
                .collect(),
        );
        assert!(!passes_filters(&event, &filter, &None));
    }

    #[test]
    fn test_passes_filters_project_match() {
        let event = make_crud_event(EntityType::Plan, Some("proj-1"));
        assert!(passes_filters(&event, &None, &Some("proj-1".to_string())));
    }

    #[test]
    fn test_passes_filters_project_no_match() {
        let event = make_crud_event(EntityType::Plan, Some("proj-2"));
        assert!(!passes_filters(&event, &None, &Some("proj-1".to_string())));
    }

    #[test]
    fn test_passes_filters_global_event_passes_project_filter() {
        // Events without project_id pass through even with a project filter
        let event = make_crud_event(EntityType::Plan, None);
        assert!(passes_filters(&event, &None, &Some("proj-1".to_string())));
    }

    #[test]
    fn test_passes_filters_combined() {
        let event = make_crud_event(EntityType::Task, Some("proj-1"));
        let entity_filter: Option<HashSet<String>> =
            Some(["task".to_string()].into_iter().collect());
        let project_filter = Some("proj-1".to_string());
        assert!(passes_filters(&event, &entity_filter, &project_filter));

        // Same entity, wrong project
        let event2 = make_crud_event(EntityType::Task, Some("proj-2"));
        assert!(!passes_filters(&event2, &entity_filter, &project_filter));

        // Wrong entity, right project
        let event3 = make_crud_event(EntityType::Note, Some("proj-1"));
        assert!(!passes_filters(&event3, &entity_filter, &project_filter));
    }

    // ================================================================
    // passes_graph_filters
    // ================================================================

    fn make_graph_event(layer: GraphLayer, project_id: &str) -> GraphEvent {
        GraphEvent::node(GraphEventType::NodeCreated, layer, "node-1", project_id)
    }

    #[test]
    fn test_graph_filters_no_filters() {
        let event = make_graph_event(GraphLayer::Knowledge, "proj-1");
        assert!(passes_graph_filters(&event, &None, &None));
    }

    #[test]
    fn test_graph_filters_layer_match() {
        let event = make_graph_event(GraphLayer::Knowledge, "proj-1");
        let filter: Option<HashSet<String>> = Some(
            ["knowledge".to_string(), "neural".to_string()]
                .into_iter()
                .collect(),
        );
        assert!(passes_graph_filters(&event, &filter, &None));
    }

    #[test]
    fn test_graph_filters_layer_no_match() {
        let event = make_graph_event(GraphLayer::Code, "proj-1");
        let filter: Option<HashSet<String>> = Some(
            ["knowledge".to_string(), "neural".to_string()]
                .into_iter()
                .collect(),
        );
        assert!(!passes_graph_filters(&event, &filter, &None));
    }

    #[test]
    fn test_graph_filters_project_match() {
        let event = make_graph_event(GraphLayer::Neural, "proj-1");
        assert!(passes_graph_filters(
            &event,
            &None,
            &Some("proj-1".to_string())
        ));
    }

    #[test]
    fn test_graph_filters_project_no_match() {
        let event = make_graph_event(GraphLayer::Neural, "proj-1");
        assert!(!passes_graph_filters(
            &event,
            &None,
            &Some("proj-2".to_string())
        ));
    }

    #[test]
    fn test_graph_filters_combined_match() {
        let event = make_graph_event(GraphLayer::Neural, "proj-1");
        let layer_filter: Option<HashSet<String>> =
            Some(["neural".to_string()].into_iter().collect());
        let project_filter = Some("proj-1".to_string());
        assert!(passes_graph_filters(&event, &layer_filter, &project_filter));
    }

    #[test]
    fn test_graph_filters_combined_layer_fail() {
        let event = make_graph_event(GraphLayer::Fabric, "proj-1");
        let layer_filter: Option<HashSet<String>> =
            Some(["neural".to_string()].into_iter().collect());
        let project_filter = Some("proj-1".to_string());
        assert!(!passes_graph_filters(
            &event,
            &layer_filter,
            &project_filter
        ));
    }

    #[test]
    fn test_graph_filters_combined_project_fail() {
        let event = make_graph_event(GraphLayer::Neural, "proj-2");
        let layer_filter: Option<HashSet<String>> =
            Some(["neural".to_string()].into_iter().collect());
        let project_filter = Some("proj-1".to_string());
        assert!(!passes_graph_filters(
            &event,
            &layer_filter,
            &project_filter
        ));
    }

    #[test]
    fn test_graph_filters_all_layers() {
        // Verify all 6 layers can be matched
        let layers = vec![
            ("code", GraphLayer::Code),
            ("knowledge", GraphLayer::Knowledge),
            ("fabric", GraphLayer::Fabric),
            ("neural", GraphLayer::Neural),
            ("skills", GraphLayer::Skills),
            ("pm", GraphLayer::Pm),
        ];
        for (name, layer) in layers {
            let event = make_graph_event(layer, "proj-1");
            let filter: Option<HashSet<String>> = Some([name.to_string()].into_iter().collect());
            assert!(
                passes_graph_filters(&event, &filter, &None),
                "Layer '{}' should pass filter",
                name
            );
        }
    }

    // ================================================================
    // WsQuery deserialization
    // ================================================================

    #[test]
    fn test_ws_query_default() {
        let query = WsQuery::default();
        assert!(query.entity_types.is_none());
        assert!(query.project_id.is_none());
        assert!(query.layers.is_none());
        assert!(query.ticket.is_none());
    }

    #[test]
    fn test_ws_query_deserialize() {
        let json =
            r#"{"entity_types":"plan,task","project_id":"proj-1","layers":"knowledge,neural"}"#;
        let query: WsQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.entity_types.as_deref(), Some("plan,task"));
        assert_eq!(query.project_id.as_deref(), Some("proj-1"));
        assert_eq!(query.layers.as_deref(), Some("knowledge,neural"));
    }

    // ================================================================
    // Batch constants
    // ================================================================

    #[test]
    fn test_batch_constants() {
        assert_eq!(GRAPH_BATCH_MAX_SIZE, 10);
        assert_eq!(GRAPH_BATCH_WINDOW, Duration::from_millis(100));
    }

    // ================================================================
    // Graph batch message format
    // ================================================================

    #[test]
    fn test_graph_batch_json_format() {
        // Verify the JSON structure that flush_graph_batch produces
        let events: Vec<GraphEvent> = vec![
            GraphEvent::node(
                GraphEventType::NodeCreated,
                GraphLayer::Knowledge,
                "note-1",
                "proj-1",
            ),
            GraphEvent::reinforcement("note-2", 0.15, "proj-1"),
        ];

        let msg = serde_json::json!({
            "kind": "graph_batch",
            "events": events,
            "count": events.len(),
        });

        let parsed: serde_json::Value = msg;
        assert_eq!(parsed["kind"], "graph_batch");
        assert_eq!(parsed["count"], 2);
        let arr = parsed["events"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["kind"], "graph");
        assert_eq!(arr[0]["type"], "node_created");
        assert_eq!(arr[0]["layer"], "knowledge");
        assert_eq!(arr[1]["type"], "reinforcement");
        assert_eq!(arr[1]["layer"], "neural");
    }

    // ================================================================
    // End-to-end graph event flow via HybridEmitter
    // ================================================================

    #[test]
    fn test_graph_events_flow_through_hybrid_emitter() {
        use crate::events::EventBus;
        use crate::events::HybridEmitter;
        use std::sync::Arc;

        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut graph_rx = hybrid.subscribe_graph();
        let mut crud_rx = hybrid.subscribe();

        // Emit a graph event via the EventEmitter trait
        use crate::events::EventEmitter;
        let emitter: Arc<dyn EventEmitter> = Arc::new(hybrid.clone());

        emitter.emit_graph(GraphEvent::node(
            GraphEventType::NodeCreated,
            GraphLayer::Knowledge,
            "note-1",
            "proj-1",
        ));

        // Graph event should arrive on graph bus
        let event = graph_rx.try_recv().unwrap();
        assert_eq!(event.event_type, GraphEventType::NodeCreated);
        assert_eq!(event.layer, GraphLayer::Knowledge);
        assert_eq!(event.node_id.as_deref(), Some("note-1"));

        // CRUD bus should be empty (graph events don't go to CRUD bus)
        assert!(crud_rx.try_recv().is_err());
    }

    #[test]
    fn test_graph_event_filter_integration() {
        // Simulate the WS filter logic: emit multiple events, filter by layer
        let events = vec![
            make_graph_event(GraphLayer::Knowledge, "proj-1"),
            make_graph_event(GraphLayer::Neural, "proj-1"),
            make_graph_event(GraphLayer::Code, "proj-1"),
            make_graph_event(GraphLayer::Fabric, "proj-2"),
        ];

        let layer_filter: Option<HashSet<String>> = Some(
            ["knowledge".to_string(), "neural".to_string()]
                .into_iter()
                .collect(),
        );
        let project_filter = Some("proj-1".to_string());

        let passed: Vec<_> = events
            .iter()
            .filter(|e| passes_graph_filters(e, &layer_filter, &project_filter))
            .collect();

        // Only knowledge + neural from proj-1 should pass
        assert_eq!(passed.len(), 2);
        assert_eq!(passed[0].layer, GraphLayer::Knowledge);
        assert_eq!(passed[1].layer, GraphLayer::Neural);
    }
}
