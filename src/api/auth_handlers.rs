//! Authentication route handlers — Google OAuth, password login, JWT token, user info.
//!
//! Endpoints:
//! - `POST /auth/login`           — Email/password login (root account + Neo4j users)
//! - `GET  /auth/google`          — Returns the Google OAuth authorization URL
//! - `POST /auth/google/callback` — Exchanges auth code for JWT + user info
//! - `GET  /auth/me`              — Returns the authenticated user (protected)
//! - `POST /auth/refresh`         — Issues a new JWT from a still-valid token (protected)

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::extractor::AuthUser;
use crate::auth::google::GoogleOAuthClient;
use crate::auth::jwt::encode_jwt;
use crate::auth::oidc::OidcClient;
use crate::neo4j::models::UserNode;
use axum::{extract::State, Json};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Request body for POST /auth/login
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

/// Request body for POST /auth/register
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
    pub name: String,
}

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

/// Request body for POST /auth/oidc/callback
#[derive(Deserialize)]
pub struct OidcCallbackRequest {
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

/// A single auth provider available for login
#[derive(Debug, Serialize)]
pub struct AuthProviderInfo {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: String,
}

/// Response for GET /auth/providers — discovery endpoint
#[derive(Debug, Serialize)]
pub struct AuthProvidersResponse {
    pub auth_required: bool,
    pub providers: Vec<AuthProviderInfo>,
    pub allow_registration: bool,
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

/// GET /auth/providers — Discovery endpoint for available auth methods.
///
/// Returns which authentication providers are configured so the frontend
/// can adapt its login UI dynamically. This endpoint is always public.
pub async fn get_auth_providers(
    State(state): State<OrchestratorState>,
) -> Json<AuthProvidersResponse> {
    let auth_config = match state.auth_config.as_ref() {
        None => {
            return Json(AuthProvidersResponse {
                auth_required: false,
                providers: vec![],
                allow_registration: false,
            });
        }
        Some(config) => config,
    };

    let mut providers = vec![];

    // Password auth (root account configured)
    if auth_config.has_password_auth() {
        providers.push(AuthProviderInfo {
            id: "password".to_string(),
            name: "Email & Password".to_string(),
            provider_type: "password".to_string(),
        });
    }

    // OIDC auth (explicit or legacy Google)
    if let Some(oidc) = auth_config.effective_oidc() {
        providers.push(AuthProviderInfo {
            id: "oidc".to_string(),
            name: oidc.provider_name.clone(),
            provider_type: "oidc".to_string(),
        });
    }

    Json(AuthProvidersResponse {
        auth_required: true,
        providers,
        allow_registration: auth_config.allow_registration,
    })
}

/// POST /auth/login — Email/password authentication.
///
/// Flow:
/// 1. Check that auth is configured and password auth is available
/// 2. Try root account first (in-memory, no DB hit)
/// 3. If not root, look up user in Neo4j by email + provider "password"
/// 4. Verify password with bcrypt
/// 5. Return JWT + user info
///
/// Security: error messages never reveal whether the email exists or not.
pub async fn password_login(
    State(state): State<OrchestratorState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<AuthTokenResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // Ensure password auth is available
    if !auth_config.has_password_auth() {
        return Err(AppError::Forbidden(
            "Password authentication is not configured".to_string(),
        ));
    }

    // Generic error to prevent user enumeration
    let invalid_credentials =
        || AppError::Unauthorized("Invalid email or password".to_string());

    // 1. Check root account first (in-memory, no DB)
    if let Some(ref root) = auth_config.root_account {
        if req.email == root.email {
            let password_ok = bcrypt::verify(&req.password, &root.password_hash)
                .unwrap_or(false);
            if password_ok {
                // Root user gets a deterministic UUID based on email
                let root_user_id =
                    Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes());
                let token = encode_jwt(
                    root_user_id,
                    &root.email,
                    &root.name,
                    &auth_config.jwt_secret,
                    auth_config.jwt_expiry_secs,
                )
                .map_err(AppError::Internal)?;

                return Ok(Json(AuthTokenResponse {
                    token,
                    user: UserResponse {
                        id: root_user_id,
                        email: root.email.clone(),
                        name: root.name.clone(),
                        picture_url: None,
                    },
                }));
            } else {
                return Err(invalid_credentials());
            }
        }
    }

    // 2. Look up user in Neo4j by email + provider "password"
    let user = state
        .orchestrator
        .neo4j()
        .get_user_by_email_and_provider(&req.email, "password")
        .await?
        .ok_or_else(invalid_credentials)?;

    // 3. Verify bcrypt password
    let password_hash = user
        .password_hash
        .as_deref()
        .ok_or_else(invalid_credentials)?;

    let password_ok =
        bcrypt::verify(&req.password, password_hash).unwrap_or(false);
    if !password_ok {
        return Err(invalid_credentials());
    }

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

/// POST /auth/register — Create a new password-authenticated account.
///
/// Only available when `allow_registration` is true in auth config.
/// Creates a user in Neo4j with bcrypt-hashed password and returns
/// JWT + user info (auto-login after registration).
pub async fn register(
    State(state): State<OrchestratorState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthTokenResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // 1. Check registration is enabled
    if !auth_config.allow_registration {
        return Err(AppError::Forbidden(
            "Registration is disabled".to_string(),
        ));
    }

    // 2. Validate input fields
    validate_registration(&req, auth_config)?;

    // 3. Check email uniqueness for password provider
    let existing = state
        .orchestrator
        .neo4j()
        .get_user_by_email_and_provider(&req.email, "password")
        .await?;
    if existing.is_some() {
        return Err(AppError::Conflict(
            "An account with this email already exists".to_string(),
        ));
    }

    // 4. Hash password with bcrypt
    let password_hash = bcrypt::hash(&req.password, 12)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to hash password: {}", e)))?;

    // 5. Create user in Neo4j
    let user = state
        .orchestrator
        .neo4j()
        .create_password_user(&req.email, &req.name, &password_hash)
        .await?;

    // 6. Generate JWT (auto-login)
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

/// Validate registration request fields.
fn validate_registration(
    req: &RegisterRequest,
    auth_config: &crate::AuthConfig,
) -> Result<(), AppError> {
    // Name must not be empty
    let name = req.name.trim();
    if name.is_empty() {
        return Err(AppError::BadRequest("Name is required".to_string()));
    }

    // Email basic validation (contains @ and a dot after @)
    let email = req.email.trim().to_lowercase();
    if !email.contains('@') || !email.split('@').nth(1).map_or(false, |d| d.contains('.')) {
        return Err(AppError::BadRequest("Invalid email format".to_string()));
    }

    // Password minimum length
    if req.password.len() < 8 {
        return Err(AppError::BadRequest(
            "Password must be at least 8 characters".to_string(),
        ));
    }

    // Check allowed email domain (if configured)
    if let Some(ref domain) = auth_config.allowed_email_domain {
        if !email.ends_with(&format!("@{}", domain)) {
            return Err(AppError::Forbidden(format!(
                "Email domain not allowed (expected @{})",
                domain
            )));
        }
    }

    Ok(())
}

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
        auth_provider: crate::neo4j::models::AuthProvider::Oidc,
        external_id: Some(google_user.google_id.clone()),
        password_hash: None,
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

/// GET /auth/oidc — Returns the OIDC authorization URL.
///
/// Uses the generic OIDC client built from `effective_oidc()` config
/// or legacy Google fields. The frontend redirects the user to this URL.
pub async fn oidc_login(
    State(state): State<OrchestratorState>,
) -> Result<Json<AuthUrlResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    if !auth_config.has_oidc() {
        return Err(AppError::Forbidden(
            "OIDC authentication is not configured".to_string(),
        ));
    }

    let client = OidcClient::from_auth_config_sync(auth_config)
        .map_err(|e| AppError::Internal(e))?;

    Ok(Json(AuthUrlResponse {
        auth_url: client.auth_url(),
    }))
}

/// POST /auth/oidc/callback — Exchange OIDC authorization code for JWT + user.
///
/// 1. Exchanges the auth code via the generic OIDC client
/// 2. Upserts the user in Neo4j with auth_provider="oidc" + external_id=sub
/// 3. Returns a JWT token + user info
pub async fn oidc_callback(
    State(state): State<OrchestratorState>,
    Json(req): Json<OidcCallbackRequest>,
) -> Result<Json<AuthTokenResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    if !auth_config.has_oidc() {
        return Err(AppError::Forbidden(
            "OIDC authentication is not configured".to_string(),
        ));
    }

    // 1. Build OIDC client and exchange code
    let client = OidcClient::from_auth_config_sync(auth_config)
        .map_err(|e| AppError::Internal(e))?;

    let oidc_user = client
        .exchange_code(&req.code)
        .await
        .map_err(|e| AppError::BadRequest(format!("OIDC code exchange failed: {}", e)))?;

    // 2. Check email domain restriction
    if let Some(ref domain) = auth_config.allowed_email_domain {
        if !oidc_user.email.ends_with(&format!("@{}", domain)) {
            return Err(AppError::Forbidden(format!(
                "Email domain not allowed (expected @{})",
                domain
            )));
        }
    }

    // 3. Upsert user in Neo4j
    let now = Utc::now();
    let user_node = UserNode {
        id: Uuid::new_v4(),
        email: oidc_user.email.clone(),
        name: oidc_user.name.clone(),
        picture_url: oidc_user.picture.clone(),
        auth_provider: crate::neo4j::models::AuthProvider::Oidc,
        external_id: Some(oidc_user.external_id.clone()),
        password_hash: None,
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
            jwt_secret: TEST_SECRET.to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: None,
            frontend_url: None,
            allow_registration: false,
            root_account: None,
            oidc: None,
            google_client_id: Some("test-id".to_string()),
            google_client_secret: Some("test-secret".to_string()),
            google_redirect_uri: Some("http://localhost:3000/auth/callback".to_string()),
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
        let public = Router::new()
            .route("/auth/providers", get(get_auth_providers))
            .route("/auth/login", post(password_login))
            .route("/auth/register", post(register))
            .route("/auth/google", get(google_login));

        // Protected auth routes (with middleware)
        let protected = Router::new()
            .route("/auth/me", get(get_me))
            .route("/auth/refresh", post(refresh_token))
            .layer(from_fn_with_state(state.clone(), require_auth));

        public.merge(protected).with_state(state)
    }

    /// Build a test router with auth routes, returning the state for further setup
    async fn test_auth_app_with_state(
        auth_config: Option<AuthConfig>,
    ) -> (Router, OrchestratorState) {
        let state = make_server_state(auth_config).await;

        let public = Router::new()
            .route("/auth/providers", get(get_auth_providers))
            .route("/auth/login", post(password_login))
            .route("/auth/register", post(register))
            .route("/auth/google", get(google_login));

        let protected = Router::new()
            .route("/auth/me", get(get_me))
            .route("/auth/refresh", post(refresh_token))
            .layer(from_fn_with_state(state.clone(), require_auth));

        let app = public.merge(protected).with_state(state.clone());
        (app, state)
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
            auth_provider: crate::neo4j::models::AuthProvider::Oidc,
            external_id: Some("google-123".to_string()),
            password_hash: None,
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
            auth_provider: crate::neo4j::models::AuthProvider::Oidc,
            external_id: Some("google-456".to_string()),
            password_hash: None,
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

    // ================================================================
    // Password login tests
    // ================================================================

    fn auth_config_with_root() -> AuthConfig {
        let password_hash = bcrypt::hash("rootpass123", 4).unwrap(); // cost 4 for fast tests
        AuthConfig {
            jwt_secret: TEST_SECRET.to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: None,
            frontend_url: None,
            allow_registration: false,
            root_account: Some(crate::RootAccountConfig {
                email: "admin@ffs.holdings".to_string(),
                name: "Admin".to_string(),
                password_hash,
            }),
            oidc: None,
            google_client_id: None,
            google_client_secret: None,
            google_redirect_uri: None,
        }
    }

    fn login_body(email: &str, password: &str) -> Body {
        Body::from(
            serde_json::json!({
                "email": email,
                "password": password
            })
            .to_string(),
        )
    }

    #[tokio::test]
    async fn test_login_root_account_success() {
        let app = test_auth_app(Some(auth_config_with_root())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("admin@ffs.holdings", "rootpass123"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["token"].as_str().is_some());
        assert_eq!(json["user"]["email"], "admin@ffs.holdings");
        assert_eq!(json["user"]["name"], "Admin");

        // Verify the token is valid
        let token = json["token"].as_str().unwrap();
        let claims = crate::auth::jwt::decode_jwt(token, TEST_SECRET).unwrap();
        assert_eq!(claims.email, "admin@ffs.holdings");
    }

    #[tokio::test]
    async fn test_login_root_account_wrong_password() {
        let app = test_auth_app(Some(auth_config_with_root())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("admin@ffs.holdings", "wrongpassword"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_login_neo4j_user_success() {
        let (app, state) = test_auth_app_with_state(Some(auth_config_with_root())).await;

        // Create a password user in the mock store
        let password_hash = bcrypt::hash("userpass456", 4).unwrap();
        state
            .orchestrator
            .neo4j()
            .create_password_user("alice@ffs.holdings", "Alice", &password_hash)
            .await
            .unwrap();

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("alice@ffs.holdings", "userpass456"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["user"]["email"], "alice@ffs.holdings");
        assert_eq!(json["user"]["name"], "Alice");
    }

    #[tokio::test]
    async fn test_login_neo4j_user_wrong_password() {
        let (app, state) = test_auth_app_with_state(Some(auth_config_with_root())).await;

        // Create a password user
        let password_hash = bcrypt::hash("correctpass", 4).unwrap();
        state
            .orchestrator
            .neo4j()
            .create_password_user("bob@ffs.holdings", "Bob", &password_hash)
            .await
            .unwrap();

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("bob@ffs.holdings", "wrongpass"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_login_unknown_email_returns_401() {
        let app = test_auth_app(Some(auth_config_with_root())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("nobody@ffs.holdings", "anything"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should be 401 (not 404) — no user enumeration
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_login_no_auth_config_returns_403() {
        let app = test_auth_app(None).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("admin@ffs.holdings", "pass"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_login_no_password_auth_returns_403() {
        // Config with OIDC only (no root account)
        let config = test_auth_config(); // has no root_account
        let app = test_auth_app(Some(config)).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/login")
            .header("content-type", "application/json")
            .body(login_body("admin@ffs.holdings", "pass"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    // ================================================================
    // GET /auth/providers tests
    // ================================================================

    async fn get_providers_json(app: Router) -> serde_json::Value {
        let req = HttpRequest::builder()
            .uri("/auth/providers")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn test_providers_no_auth_config() {
        let app = test_auth_app(None).await;
        let json = get_providers_json(app).await;

        assert_eq!(json["auth_required"], false);
        assert_eq!(json["providers"].as_array().unwrap().len(), 0);
        assert_eq!(json["allow_registration"], false);
    }

    #[tokio::test]
    async fn test_providers_password_only() {
        let config = auth_config_with_root();
        let app = test_auth_app(Some(config)).await;
        let json = get_providers_json(app).await;

        assert_eq!(json["auth_required"], true);
        let providers = json["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0]["id"], "password");
        assert_eq!(providers[0]["type"], "password");
        assert_eq!(providers[0]["name"], "Email & Password");
        assert_eq!(json["allow_registration"], false);
    }

    #[tokio::test]
    async fn test_providers_oidc_only() {
        // test_auth_config has legacy google_* fields → effective_oidc() returns Some
        let config = test_auth_config();
        let app = test_auth_app(Some(config)).await;
        let json = get_providers_json(app).await;

        assert_eq!(json["auth_required"], true);
        let providers = json["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0]["id"], "oidc");
        assert_eq!(providers[0]["type"], "oidc");
        // Legacy google fields → provider_name defaults to "Google"
        assert_eq!(providers[0]["name"], "Google");
    }

    #[tokio::test]
    async fn test_providers_both_password_and_oidc() {
        let mut config = auth_config_with_root();
        // Add legacy Google OIDC
        config.google_client_id = Some("test-id".to_string());
        config.google_client_secret = Some("test-secret".to_string());
        config.google_redirect_uri = Some("http://localhost/callback".to_string());

        let app = test_auth_app(Some(config)).await;
        let json = get_providers_json(app).await;

        assert_eq!(json["auth_required"], true);
        let providers = json["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 2);

        let types: Vec<&str> = providers.iter().map(|p| p["type"].as_str().unwrap()).collect();
        assert!(types.contains(&"password"));
        assert!(types.contains(&"oidc"));
    }

    #[tokio::test]
    async fn test_providers_allow_registration_true() {
        let mut config = auth_config_with_root();
        config.allow_registration = true;

        let app = test_auth_app(Some(config)).await;
        let json = get_providers_json(app).await;

        assert_eq!(json["allow_registration"], true);
    }

    // ================================================================
    // POST /auth/register tests
    // ================================================================

    fn auth_config_with_registration() -> AuthConfig {
        let mut config = auth_config_with_root();
        config.allow_registration = true;
        config
    }

    fn register_body(email: &str, password: &str, name: &str) -> Body {
        Body::from(
            serde_json::json!({
                "email": email,
                "password": password,
                "name": name
            })
            .to_string(),
        )
    }

    #[tokio::test]
    async fn test_register_success() {
        let app = test_auth_app(Some(auth_config_with_registration())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("newuser@ffs.holdings", "securepass123", "New User"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["token"].as_str().is_some());
        assert_eq!(json["user"]["email"], "newuser@ffs.holdings");
        assert_eq!(json["user"]["name"], "New User");

        // Verify the token is valid
        let token = json["token"].as_str().unwrap();
        let claims = crate::auth::jwt::decode_jwt(token, TEST_SECRET).unwrap();
        assert_eq!(claims.email, "newuser@ffs.holdings");
    }

    #[tokio::test]
    async fn test_register_disabled_returns_403() {
        // allow_registration = false (default)
        let app = test_auth_app(Some(auth_config_with_root())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("newuser@ffs.holdings", "securepass123", "New User"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_register_duplicate_email_returns_409() {
        let (app, state) =
            test_auth_app_with_state(Some(auth_config_with_registration())).await;

        // Pre-create a password user
        let password_hash = bcrypt::hash("existing123", 4).unwrap();
        state
            .orchestrator
            .neo4j()
            .create_password_user("existing@ffs.holdings", "Existing", &password_hash)
            .await
            .unwrap();

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("existing@ffs.holdings", "newpass123", "Another"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_register_invalid_email() {
        let app = test_auth_app(Some(auth_config_with_registration())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("not-an-email", "securepass123", "User"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_register_password_too_short() {
        let app = test_auth_app(Some(auth_config_with_registration())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("user@ffs.holdings", "short", "User"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_register_empty_name() {
        let app = test_auth_app(Some(auth_config_with_registration())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("user@ffs.holdings", "securepass123", "  "))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_register_wrong_domain() {
        let mut config = auth_config_with_registration();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_auth_app(Some(config)).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/register")
            .header("content-type", "application/json")
            .body(register_body("user@gmail.com", "securepass123", "User"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_providers_explicit_oidc_config() {
        let config = AuthConfig {
            jwt_secret: TEST_SECRET.to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: None,
            frontend_url: None,
            allow_registration: false,
            root_account: None,
            oidc: Some(crate::OidcConfig {
                provider_name: "Okta".to_string(),
                client_id: "okta-id".to_string(),
                client_secret: "okta-secret".to_string(),
                redirect_uri: "http://localhost/callback".to_string(),
                auth_endpoint: Some("https://okta.example.com/authorize".to_string()),
                token_endpoint: Some("https://okta.example.com/token".to_string()),
                userinfo_endpoint: Some("https://okta.example.com/userinfo".to_string()),
                scopes: "openid email profile".to_string(),
                discovery_url: None,
            }),
            google_client_id: None,
            google_client_secret: None,
            google_redirect_uri: None,
        };

        let app = test_auth_app(Some(config)).await;
        let json = get_providers_json(app).await;

        let providers = json["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0]["name"], "Okta");
        assert_eq!(providers[0]["type"], "oidc");
    }
}
