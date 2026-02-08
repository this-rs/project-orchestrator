//! Chat API handlers — SSE streaming + session management

use crate::api::handlers::{AppError, OrchestratorState};
use crate::api::query::{PaginatedResponse, PaginationParams};
use crate::chat::types::{ChatRequest, ChatSession, ClientMessage, CreateSessionResponse};
use axum::{
    extract::{Path, Query, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures::stream::Stream;
use serde::Deserialize;
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

// ============================================================================
// Create session + send first message
// ============================================================================

/// POST /api/chat/sessions — Create a new chat session and send the first message
pub async fn create_session(
    State(state): State<OrchestratorState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let response = chat_manager
        .create_session(&request)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(response))
}

// ============================================================================
// SSE stream
// ============================================================================

/// GET /api/chat/sessions/{id}/stream — Subscribe to SSE event stream
pub async fn stream_events(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let rx = chat_manager
        .subscribe(&session_id)
        .await
        .map_err(|e| AppError::NotFound(format!("Session not found: {}", e)))?;

    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let json = serde_json::to_string(&event).unwrap_or_default();
            Some(Ok(Event::default().event(event.event_type()).data(json)))
        }
        Err(_) => None, // Lagged — skip missed events
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

// ============================================================================
// Send follow-up message
// ============================================================================

/// POST /api/chat/sessions/{id}/messages — Send a follow-up message
pub async fn send_message(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
    Json(client_msg): Json<ClientMessage>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    // Extract the message text based on ClientMessage type
    let message = match &client_msg {
        ClientMessage::UserMessage { content } => content.clone(),
        ClientMessage::PermissionResponse { allow, reason } => {
            // For now, format as text — the CLI handles permissions via control protocol
            format!(
                "Permission {}: {}",
                if *allow { "granted" } else { "denied" },
                reason.as_deref().unwrap_or("")
            )
        }
        ClientMessage::InputResponse { content } => content.clone(),
    };

    // Check if session is active; if not, auto-resume
    if !chat_manager.is_session_active(&session_id).await {
        chat_manager
            .resume_session(&session_id, &message)
            .await
            .map_err(AppError::Internal)?;
    } else {
        chat_manager
            .send_message(&session_id, &message)
            .await
            .map_err(AppError::Internal)?;
    }

    Ok(Json(serde_json::json!({ "status": "sent" })))
}

// ============================================================================
// Interrupt
// ============================================================================

/// POST /api/chat/sessions/{id}/interrupt — Interrupt current operation
pub async fn interrupt_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    chat_manager
        .interrupt(&session_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "status": "interrupted" })))
}

// ============================================================================
// Message history
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct MessagesQuery {
    #[serde(default = "default_messages_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_messages_limit() -> usize {
    50
}

/// GET /api/chat/sessions/{id}/messages — Get message history
pub async fn list_messages(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
    Query(query): Query<MessagesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let loaded = chat_manager
        .get_session_messages(
            &session_id.to_string(),
            Some(query.limit),
            Some(query.offset),
        )
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("no conversation_id") {
                AppError::NotFound(msg)
            } else {
                AppError::Internal(e)
            }
        })?;

    // Convert to chronological order for UI display
    let messages: Vec<serde_json::Value> = loaded
        .messages_chronological()
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "conversation_id": m.conversation_id,
                "role": m.role,
                "content": m.content,
                "turn_index": m.turn_index,
                "created_at": m.created_at,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "messages": messages,
        "total_count": loaded.total_count,
        "has_more": loaded.has_more,
        "offset": loaded.offset,
        "limit": loaded.limit,
    })))
}

// ============================================================================
// Session CRUD
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SessionsListQuery {
    #[serde(default)]
    pub project_slug: Option<String>,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

/// GET /api/chat/sessions — List chat sessions
pub async fn list_sessions(
    State(state): State<OrchestratorState>,
    Query(query): Query<SessionsListQuery>,
) -> Result<Json<PaginatedResponse<ChatSession>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let (sessions, total) = state
        .orchestrator
        .neo4j()
        .list_chat_sessions(
            query.project_slug.as_deref(),
            query.pagination.validated_limit(),
            query.pagination.offset,
        )
        .await
        .map_err(AppError::Internal)?;

    // Convert ChatSessionNode → ChatSession
    let items: Vec<ChatSession> = sessions
        .into_iter()
        .map(|s| ChatSession {
            id: s.id.to_string(),
            cli_session_id: s.cli_session_id,
            project_slug: s.project_slug,
            cwd: s.cwd,
            title: s.title,
            model: s.model,
            created_at: s.created_at.to_rfc3339(),
            updated_at: s.updated_at.to_rfc3339(),
            message_count: s.message_count,
            total_cost_usd: s.total_cost_usd,
            conversation_id: s.conversation_id,
        })
        .collect();

    Ok(Json(PaginatedResponse::new(
        items,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// GET /api/chat/sessions/{id} — Get session details
pub async fn get_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<ChatSession>, AppError> {
    let node = state
        .orchestrator
        .neo4j()
        .get_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Session {} not found", session_id)))?;

    Ok(Json(ChatSession {
        id: node.id.to_string(),
        cli_session_id: node.cli_session_id,
        project_slug: node.project_slug,
        cwd: node.cwd,
        title: node.title,
        model: node.model,
        created_at: node.created_at.to_rfc3339(),
        updated_at: node.updated_at.to_rfc3339(),
        message_count: node.message_count,
        total_cost_usd: node.total_cost_usd,
        conversation_id: node.conversation_id,
    }))
}

/// DELETE /api/chat/sessions/{id} — Delete a session
pub async fn delete_session(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Close active session if running
    if let Some(chat_manager) = &state.chat_manager {
        let _ = chat_manager.close_session(&session_id.to_string()).await;
    }

    // Delete from Neo4j
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_chat_session(session_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(Json(serde_json::json!({ "deleted": true })))
    } else {
        Err(AppError::NotFound(format!(
            "Session {} not found",
            session_id
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_chat_session};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Build an OrchestratorState with mock backends (no ChatManager)
    async fn mock_server_state() -> OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::EventBus::default()),
        })
    }

    /// Build a test router with mock state
    async fn test_app() -> axum::Router {
        let state = mock_server_state().await;
        create_router(state)
    }

    /// Build a test router with pre-seeded sessions
    async fn test_app_with_sessions(
        sessions: &[crate::neo4j::models::ChatSessionNode],
    ) -> axum::Router {
        let app_state = mock_app_state();
        for s in sessions {
            app_state.neo4j.create_chat_session(s).await.unwrap();
        }
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::EventBus::default()),
        });
        create_router(state)
    }

    // ====================================================================
    // SessionsListQuery serde
    // ====================================================================

    #[test]
    fn test_sessions_list_query_defaults() {
        let json = r#"{}"#;
        let query: SessionsListQuery = serde_json::from_str(json).unwrap();
        assert!(query.project_slug.is_none());
    }

    #[test]
    fn test_sessions_list_query_with_project() {
        let json = r#"{"project_slug": "my-project"}"#;
        let query: SessionsListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.project_slug.as_deref(), Some("my-project"));
    }

    // ====================================================================
    // GET /api/chat/sessions — list
    // ====================================================================

    #[tokio::test]
    async fn test_list_sessions_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chat/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_sessions_with_data() {
        let s1 = test_chat_session(Some("proj-a"));
        let s2 = test_chat_session(Some("proj-b"));
        let app = test_app_with_sessions(&[s1, s2]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chat/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 2);
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_list_sessions_filter_by_project() {
        let s1 = test_chat_session(Some("proj-a"));
        let s2 = test_chat_session(Some("proj-a"));
        let s3 = test_chat_session(Some("proj-b"));
        let app = test_app_with_sessions(&[s1, s2, s3]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chat/sessions?project_slug=proj-a")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 2);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id} — get
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_found() {
        let session = test_chat_session(Some("my-proj"));
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/chat/sessions/{}", session_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], session_id.to_string());
        assert_eq!(json["project_slug"], "my-proj");
        assert_eq!(json["model"], "claude-opus-4-6");
    }

    #[tokio::test]
    async fn test_get_session_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/chat/sessions/{}", fake_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // DELETE /api/chat/sessions/{id} — delete
    // ====================================================================

    #[tokio::test]
    async fn test_delete_session_found() {
        let session = test_chat_session(None);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(&format!("/api/chat/sessions/{}", session_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["deleted"], true);
    }

    #[tokio::test]
    async fn test_delete_session_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(&format!("/api/chat/sessions/{}", fake_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // POST endpoints — no chat_manager returns error
    // ====================================================================

    #[tokio::test]
    async fn test_create_session_no_chat_manager() {
        let app = test_app().await;

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/sessions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"message":"Hello","cwd":"/tmp"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_send_message_no_chat_manager() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/chat/sessions/{}/messages", fake_id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"user_message","content":"Hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_interrupt_no_chat_manager() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/chat/sessions/{}/interrupt", fake_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_stream_no_chat_manager() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/chat/sessions/{}/stream", fake_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
