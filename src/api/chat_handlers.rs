//! Chat API handlers — session management (streaming via WebSocket)

use crate::api::handlers::{AppError, OrchestratorState};
use crate::api::query::{PaginatedResponse, PaginationParams};
use crate::chat::types::{ChatRequest, ChatSession, CreateSessionResponse, MessageSearchResult};
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
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

    // Emit CRUD event for live refresh
    state.event_bus.emit(
        CrudEvent::new(
            EntityType::ChatSession,
            CrudAction::Created,
            &response.session_id,
        )
        .with_payload(serde_json::json!({
            "project_slug": request.project_slug,
        })),
    );

    Ok(Json(response))
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

    // Messages are already sorted by created_at:asc from Meilisearch (see manager)
    let messages: Vec<serde_json::Value> = loaded
        .messages
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
            preview: s.preview,
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
        preview: node.preview,
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
        // Emit CRUD event for live refresh
        state.event_bus.emit(CrudEvent::new(
            EntityType::ChatSession,
            CrudAction::Deleted,
            session_id.to_string(),
        ));

        Ok(Json(serde_json::json!({ "deleted": true })))
    } else {
        Err(AppError::NotFound(format!(
            "Session {} not found",
            session_id
        )))
    }
}

// ============================================================================
// Search messages across sessions
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SearchMessagesQuery {
    /// Full-text search query
    pub q: String,
    /// Optional project slug filter
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Maximum number of session groups to return (default 10)
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    10
}

/// GET /api/chat/search — Search messages across all sessions
pub async fn search_messages(
    State(state): State<OrchestratorState>,
    Query(query): Query<SearchMessagesQuery>,
) -> Result<Json<Vec<MessageSearchResult>>, AppError> {
    if query.q.trim().is_empty() {
        return Err(AppError::BadRequest(
            "Search query 'q' cannot be empty".to_string(),
        ));
    }

    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let results = chat_manager
        .search_messages(&query.q, query.limit, query.project_slug.as_deref())
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(results))
}

// ============================================================================
// Backfill
// ============================================================================

/// POST /api/chat/sessions/backfill-previews — Backfill title/preview for existing sessions
pub async fn backfill_previews(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Phase 1: backfill from Neo4j events (fast, for sessions with stored events)
    let neo4j_count = state
        .orchestrator
        .neo4j()
        .backfill_chat_session_previews()
        .await
        .map_err(AppError::Internal)?;

    // Phase 2: backfill from Meilisearch (for older sessions without Neo4j events)
    let meili_count = if let Some(chat_manager) = &state.chat_manager {
        chat_manager
            .backfill_previews_from_meilisearch()
            .await
            .unwrap_or(0)
    } else {
        0
    };

    let total = neo4j_count + meili_count;
    Ok(Json(serde_json::json!({
        "updated": total,
        "from_neo4j": neo4j_count,
        "from_meilisearch": meili_count,
        "message": format!("Backfilled title/preview for {} sessions", total)
    })))
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
            auth_config: None,
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
            auth_config: None,
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
                    .uri(format!("/api/chat/sessions/{}", session_id))
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
                    .uri(format!("/api/chat/sessions/{}", fake_id))
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
                    .uri(format!("/api/chat/sessions/{}", session_id))
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
                    .uri(format!("/api/chat/sessions/{}", fake_id))
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

    // ====================================================================
    // MessagesQuery serde
    // ====================================================================

    #[test]
    fn test_messages_query_defaults() {
        let json = r#"{}"#;
        let query: MessagesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.limit, 50);
        assert_eq!(query.offset, 0);
    }

    #[test]
    fn test_messages_query_custom() {
        let json = r#"{"limit": 10, "offset": 5}"#;
        let query: MessagesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.limit, 10);
        assert_eq!(query.offset, 5);
    }

    #[test]
    fn test_default_messages_limit_value() {
        assert_eq!(default_messages_limit(), 50);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id}/messages — no chat_manager
    // ====================================================================

    #[tokio::test]
    async fn test_list_messages_no_chat_manager() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/chat/sessions/{}/messages", fake_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // No chat_manager → 500 Internal Server Error
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_list_messages_invalid_session_id() {
        let app = test_app().await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/chat/sessions/not-a-uuid/messages")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Invalid UUID in path → 400 Bad Request
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // GET /api/chat/sessions/{id} — conversation_id field
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_includes_conversation_id() {
        let mut session = test_chat_session(Some("my-proj"));
        session.conversation_id = Some("conv-test-123".into());
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/chat/sessions/{}", session_id))
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
        assert_eq!(json["conversation_id"], "conv-test-123");
    }

    #[tokio::test]
    async fn test_get_session_conversation_id_null() {
        let session = test_chat_session(None);
        let session_id = session.id;
        let app = test_app_with_sessions(&[session]).await;

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/chat/sessions/{}", session_id))
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
        assert!(json["conversation_id"].is_null());
    }

    // ====================================================================
    // GET /api/chat/sessions — list includes conversation_id
    // ====================================================================

    #[tokio::test]
    async fn test_list_sessions_includes_conversation_id() {
        let mut session = test_chat_session(Some("proj-a"));
        session.conversation_id = Some("conv-xyz".into());
        let app = test_app_with_sessions(&[session]).await;

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
        assert_eq!(json["items"][0]["conversation_id"], "conv-xyz");
    }
}
