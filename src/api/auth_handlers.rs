//! Authentication route handlers — Google OAuth, JWT token, user info.
//!
//! Endpoints:
//! - `GET  /auth/google`          — Returns the Google OAuth authorization URL
//! - `POST /auth/google/callback` — Exchanges auth code for JWT + user info
//! - `GET  /auth/me`              — Returns the authenticated user (protected)
//! - `POST /auth/refresh`         — Issues a new JWT from a still-valid token (protected)

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::extractor::AuthUser;
use crate::auth::google::GoogleOAuthClient;
use crate::auth::jwt::encode_jwt;
use crate::neo4j::models::UserNode;
use axum::{extract::State, Json};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Response for GET /auth/google
#[derive(Serialize)]
pub struct AuthUrlResponse {
    pub auth_url: String,
}

/// Request body for POST /auth/google/callback
#[derive(Deserialize)]
pub struct GoogleCallbackRequest {
    pub code: String,
}

/// Response for POST /auth/google/callback
#[derive(Serialize)]
pub struct AuthTokenResponse {
    pub token: String,
    pub user: UserResponse,
}

/// Response for POST /auth/refresh
#[derive(Serialize)]
pub struct RefreshTokenResponse {
    pub token: String,
}

/// Public user info (safe to send to client)
#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub picture_url: Option<String>,
}

impl From<UserNode> for UserResponse {
    fn from(u: UserNode) -> Self {
        Self {
            id: u.id,
            email: u.email,
            name: u.name,
            picture_url: u.picture_url,
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /auth/google — Returns the Google OAuth authorization URL.
///
/// The frontend redirects the user to this URL to start the OAuth flow.
pub async fn google_login(
    State(state): State<OrchestratorState>,
) -> Result<Json<AuthUrlResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    let client = GoogleOAuthClient::new(auth_config);
    let auth_url = client.auth_url();

    Ok(Json(AuthUrlResponse { auth_url }))
}

/// POST /auth/google/callback — Exchange authorization code for JWT + user.
///
/// 1. Exchanges the Google auth code for user info
/// 2. Upserts the user in Neo4j (creates on first login, updates on subsequent)
/// 3. Returns a JWT token + user info
pub async fn google_callback(
    State(state): State<OrchestratorState>,
    Json(req): Json<GoogleCallbackRequest>,
) -> Result<Json<AuthTokenResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // 1. Exchange code for Google user info
    let client = GoogleOAuthClient::new(auth_config);
    let google_user = client
        .exchange_code(&req.code)
        .await
        .map_err(|e| AppError::BadRequest(format!("OAuth code exchange failed: {}", e)))?;

    // 2. Check email domain restriction (if configured)
    if let Some(ref domain) = auth_config.allowed_email_domain {
        if !google_user.email.ends_with(&format!("@{}", domain)) {
            return Err(AppError::Forbidden(format!(
                "Email domain not allowed (expected @{})",
                domain
            )));
        }
    }

    // 3. Upsert user in Neo4j
    let now = Utc::now();
    let user_node = UserNode {
        id: Uuid::new_v4(), // Will be overridden by MERGE if existing
        email: google_user.email.clone(),
        name: google_user.name.clone(),
        picture_url: google_user.picture.clone(),
        google_id: google_user.google_id.clone(),
        created_at: now,
        last_login_at: now,
    };

    let user = state.orchestrator.neo4j().upsert_user(&user_node).await?;

    // 4. Generate JWT
    let token = encode_jwt(
        user.id,
        &user.email,
        &user.name,
        &auth_config.jwt_secret,
        auth_config.jwt_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    Ok(Json(AuthTokenResponse {
        token,
        user: UserResponse::from(user),
    }))
}

/// GET /auth/me — Returns the authenticated user's info.
///
/// Requires a valid JWT token (via `require_auth` middleware + `AuthUser` extractor).
pub async fn get_me(
    State(state): State<OrchestratorState>,
    user: AuthUser,
) -> Result<Json<UserResponse>, AppError> {
    // Fetch full user from Neo4j (may have updated picture, etc.)
    let user_node = state
        .orchestrator
        .neo4j()
        .get_user_by_id(user.user_id)
        .await?
        .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    Ok(Json(UserResponse::from(user_node)))
}

/// POST /auth/refresh — Issue a new JWT from a still-valid token.
///
/// The caller must present a valid (non-expired) Bearer token.
/// Returns a fresh token with a new 8h expiry window.
/// This is independent of WebSocket connections — a WS stream can
/// continue uninterrupted while the client refreshes the token via HTTP.
pub async fn refresh_token(
    State(state): State<OrchestratorState>,
    user: AuthUser,
) -> Result<Json<RefreshTokenResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    let token = encode_jwt(
        user.user_id,
        &user.email,
        &user.name,
        &auth_config.jwt_secret,
        auth_config.jwt_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    Ok(Json(RefreshTokenResponse { token }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::jwt::encode_jwt;
    use crate::auth::middleware::require_auth;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::mock_app_state;
    use crate::AuthConfig;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::routing::{get, post};
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // oneshot

    const TEST_SECRET: &str = "test-secret-key-minimum-32-chars!!";

    fn test_auth_config() -> AuthConfig {
        AuthConfig {
            google_client_id: "test-id".to_string(),
            google_client_secret: "test-secret".to_string(),
            google_redirect_uri: "http://localhost:3000/auth/callback".to_string(),
            jwt_secret: TEST_SECRET.to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: None,
            frontend_url: None,
        }
    }

    async fn make_server_state(auth_config: Option<AuthConfig>) -> OrchestratorState {
        let state = mock_app_state();
        let event_bus = Arc::new(EventBus::default());
        let orchestrator = Arc::new(
            Orchestrator::with_event_bus(state, event_bus.clone())
                .await
                .unwrap(),
        );
        let watcher = FileWatcher::new(orchestrator.clone());

        Arc::new(crate::api::handlers::ServerState {
            orchestrator,
            watcher: Arc::new(RwLock::new(watcher)),
            chat_manager: None,
            event_bus,
            auth_config,
        })
    }

    /// Build a test router with auth routes (some protected, some public)
    async fn test_auth_app(auth_config: Option<AuthConfig>) -> Router {
        let state = make_server_state(auth_config).await;

        // Public auth routes (no middleware)
        let public = Router::new().route("/auth/google", get(google_login));

        // Protected auth routes (with middleware)
        let protected = Router::new()
            .route("/auth/me", get(get_me))
            .route("/auth/refresh", post(refresh_token))
            .layer(from_fn_with_state(state.clone(), require_auth));

        public.merge(protected).with_state(state)
    }

    #[tokio::test]
    async fn test_google_login_returns_auth_url() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/auth/google")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let auth_url = json["auth_url"].as_str().unwrap();

        assert!(auth_url.starts_with("https://accounts.google.com/o/oauth2/v2/auth"));
        assert!(auth_url.contains("client_id=test-id"));
        assert!(auth_url.contains("response_type=code"));
    }

    #[tokio::test]
    async fn test_google_login_no_auth_config_returns_403() {
        let app = test_auth_app(None).await;

        let req = HttpRequest::builder()
            .uri("/auth/google")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_get_me_without_token_returns_401_or_403() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/auth/me")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Without token, the middleware should reject (401 Unauthorized)
        assert!(
            resp.status() == StatusCode::UNAUTHORIZED || resp.status() == StatusCode::FORBIDDEN,
            "Expected 401 or 403, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_get_me_with_valid_token() {
        let state = make_server_state(Some(test_auth_config())).await;

        // First, create a user in the mock store
        let user_id = Uuid::new_v4();
        let now = Utc::now();
        let user_node = UserNode {
            id: user_id,
            email: "alice@ffs.holdings".to_string(),
            name: "Alice".to_string(),
            picture_url: Some("https://example.com/photo.jpg".to_string()),
            google_id: "google-123".to_string(),
            created_at: now,
            last_login_at: now,
        };
        state
            .orchestrator
            .neo4j()
            .upsert_user(&user_node)
            .await
            .unwrap();

        // Generate a valid token for this user
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600).unwrap();

        // Build app with same state
        let public = Router::new().route("/auth/google", get(google_login));
        let protected = Router::new()
            .route("/auth/me", get(get_me))
            .route("/auth/refresh", post(refresh_token))
            .layer(from_fn_with_state(state.clone(), require_auth));
        let app = public.merge(protected).with_state(state);

        let req = HttpRequest::builder()
            .uri("/auth/me")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["email"], "alice@ffs.holdings");
        assert_eq!(json["name"], "Alice");
        assert_eq!(json["id"], user_id.to_string());
    }

    #[tokio::test]
    async fn test_refresh_returns_new_token() {
        let state = make_server_state(Some(test_auth_config())).await;

        // Create user
        let user_id = Uuid::new_v4();
        let now = Utc::now();
        let user_node = UserNode {
            id: user_id,
            email: "bob@ffs.holdings".to_string(),
            name: "Bob".to_string(),
            picture_url: None,
            google_id: "google-456".to_string(),
            created_at: now,
            last_login_at: now,
        };
        state
            .orchestrator
            .neo4j()
            .upsert_user(&user_node)
            .await
            .unwrap();

        // Generate initial token
        let token = encode_jwt(user_id, "bob@ffs.holdings", "Bob", TEST_SECRET, 3600).unwrap();

        // Build app
        let protected = Router::new()
            .route("/auth/refresh", post(refresh_token))
            .layer(from_fn_with_state(state.clone(), require_auth));
        let app = protected.with_state(state);

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let new_token = json["token"].as_str().unwrap();

        // Verify the new token is valid and different
        assert!(!new_token.is_empty());
        assert_ne!(new_token, token);

        // Decode to verify claims
        let claims = crate::auth::jwt::decode_jwt(new_token, TEST_SECRET).unwrap();
        assert_eq!(claims.sub, user_id.to_string());
        assert_eq!(claims.email, "bob@ffs.holdings");
    }
}
