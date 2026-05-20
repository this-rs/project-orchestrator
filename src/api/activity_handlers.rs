//! REST handlers for the **Live Activity Hub** snapshot endpoint.
//!
//! `GET /api/activity/snapshot?project_id=<uuid>[&project_slug=<slug>][&chat_limit=<n>]`
//!
//! Returns an [`ActivitySnapshot`] capturing the current state of all in-flight
//! work for a project (PlanRuns + ProtocolRuns + ChatSessions) along with the
//! current global event sequence head. The frontend calls this on initial load
//! and after a WebSocket reconnect (with the `last_event_seq` from the previous
//! `/ws/activity` session) to avoid the "empty screen while waiting" problem
//! described in the project-orchestrator gotchas.
//!
//! See [`crate::api::models::activity`] for the response schema and
//! [`crate::neo4j::activity_snapshot`] for the Neo4j query that backs it.

use super::handlers::OrchestratorState;
use crate::api::models::activity::ActivitySnapshot;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};
use uuid::Uuid;

/// Default cap on the number of `ChatSession` rows returned in the snapshot.
///
/// Picked to comfortably fit a sidebar list while keeping the payload small
/// (≈ 50 × ~300B ≈ 15 KB). Override via the `chat_limit` query param.
const DEFAULT_CHAT_LIMIT: i64 = 50;

/// Hard cap that the handler will refuse to exceed, even if the client asks
/// for more. Protects against accidental fanout on enormous projects.
const MAX_CHAT_LIMIT: i64 = 200;

/// Query parameters for `GET /api/activity/snapshot`.
#[derive(Debug, Deserialize)]
pub struct ActivitySnapshotParams {
    /// Project UUID — required. Scope of every row in the snapshot.
    pub project_id: Uuid,
    /// Optional project slug. When omitted the handler resolves it from the
    /// project node; passing it explicitly saves one Neo4j roundtrip.
    pub project_slug: Option<String>,
    /// Optional override for the number of chat sessions returned.
    pub chat_limit: Option<i64>,
}

/// `GET /api/activity/snapshot` — return the current Activity Hub state.
pub async fn get_activity_snapshot(
    State(state): State<OrchestratorState>,
    Query(params): Query<ActivitySnapshotParams>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let started = Instant::now();

    let chat_limit = params
        .chat_limit
        .unwrap_or(DEFAULT_CHAT_LIMIT)
        .clamp(0, MAX_CHAT_LIMIT);

    // Resolve project slug (used for the ChatSession query) — either from the
    // request or by looking up the project node.
    let project_slug = match params.project_slug {
        Some(s) if !s.trim().is_empty() => Some(s),
        _ => {
            let graph = state.orchestrator.neo4j_arc();
            match graph.get_project(params.project_id).await {
                Ok(Some(p)) => Some(p.slug),
                Ok(None) => {
                    warn!(project_id = %params.project_id, "activity snapshot: project not found");
                    return Err((
                        StatusCode::NOT_FOUND,
                        format!("project {} not found", params.project_id),
                    ));
                }
                Err(e) => {
                    warn!(error = %e, "activity snapshot: failed to resolve project slug");
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to resolve project".into(),
                    ));
                }
            }
        }
    };

    let graph = state.orchestrator.neo4j_arc();

    let data = match graph
        .fetch_activity_snapshot(params.project_id, project_slug.as_deref(), chat_limit)
        .await
    {
        Ok(d) => d,
        Err(e) => {
            warn!(error = %e, project_id = %params.project_id, "activity snapshot: query failed");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "snapshot query failed".into(),
            ));
        }
    };

    let snapshot = ActivitySnapshot {
        plan_runs: data.plan_runs,
        protocol_runs: data.protocol_runs,
        chat_sessions: data.chat_sessions,
        last_event_seq: state.event_bus.local_bus().current_sequence(),
    };

    let elapsed_ms = started.elapsed().as_millis();
    debug!(
        project_id = %params.project_id,
        plan_runs = snapshot.plan_runs.len(),
        protocol_runs = snapshot.protocol_runs.len(),
        chat_sessions = snapshot.chat_sessions.len(),
        last_event_seq = snapshot.last_event_seq,
        elapsed_ms,
        "activity snapshot served"
    );

    Ok(Json(snapshot))
}

// Re-export so call sites in routes.rs don't need to know about the
// `Arc`-typed handler signature.
pub use self::get_activity_snapshot as snapshot_handler;

// Tiny type re-export to keep the public surface tight.
#[allow(dead_code)]
pub(crate) type SharedState = Arc<crate::api::handlers::ServerState>;
