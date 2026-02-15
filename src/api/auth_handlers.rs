//! Authentication route handlers — Google OAuth, password login, JWT token, user info.
//!
//! Endpoints:
//! - `POST /auth/login`           — Email/password login (root account + Neo4j users)
//! - `GET  /auth/google`          — Returns the Google OAuth authorization URL
//! - `POST /auth/google/callback` — Exchanges auth code for JWT + user info
//! - `GET  /auth/me`              — Returns the authenticated user (protected)
//! - `POST /auth/refresh`         — Issues a new access JWT from a valid refresh cookie
//! - `POST /auth/logout`          — Revokes the refresh token and clears the cookie

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::extractor::AuthUser;
use crate::auth::google::GoogleOAuthClient;
use crate::auth::jwt::encode_jwt;
use crate::auth::oidc::OidcClient;
use crate::auth::refresh;
use crate::neo4j::models::UserNode;
use crate::AuthConfig;
use axum::{
    extract::{Query as AxumQuery, State},
    http::header::SET_COOKIE,
    response::{IntoResponse, Response},
    Json,
};
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

/// Query parameters for GET /auth/google and GET /auth/oidc
///
/// The `origin` param allows the frontend to send its current `window.location.origin`
/// so the backend can build the correct `redirect_uri` for dual-access setups
/// (desktop via localhost + web via public domain).
#[derive(Debug, Deserialize)]
pub struct OAuthLoginQuery {
    pub origin: Option<String>,
}

/// Request body for POST /auth/google/callback
#[derive(Deserialize)]
pub struct GoogleCallbackRequest {
    pub code: String,
    /// The origin used when initiating the OAuth flow (must match for token exchange).
    pub origin: Option<String>,
}

/// Request body for POST /auth/oidc/callback
#[derive(Deserialize)]
pub struct OidcCallbackRequest {
    pub code: String,
    /// The origin used when initiating the OAuth flow (must match for token exchange).
    pub origin: Option<String>,
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
    /// True when this user is the root account (configured in config.yaml).
    /// Used by the frontend to gate access to the setup wizard reconfiguration.
    pub is_root: bool,
}

impl From<UserNode> for UserResponse {
    fn from(u: UserNode) -> Self {
        Self {
            id: u.id,
            email: u.email,
            name: u.name,
            picture_url: u.picture_url,
            is_root: false, // Neo4j users are never root
        }
    }
}

// ============================================================================
// Refresh token helpers
// ============================================================================

/// Generate a refresh token, store its hash in the database, and return a
/// `Set-Cookie` header alongside the usual JSON response.
///
/// This is called by every login-like handler (password_login, register,
/// google_callback, oidc_callback) to emit the HttpOnly cookie.
async fn create_refresh_token_and_cookie(
    state: &OrchestratorState,
    auth_config: &AuthConfig,
    user_id: Uuid,
) -> Result<axum::http::HeaderValue, AppError> {
    let raw_token = refresh::generate_token();
    let token_hash = refresh::hash_token(&raw_token);
    let expires_at = Utc::now()
        + chrono::Duration::seconds(auth_config.refresh_token_expiry_secs as i64);

    state
        .orchestrator
        .neo4j()
        .create_refresh_token(user_id, &token_hash, expires_at)
        .await?;

    let is_secure = refresh::should_set_secure(state.public_url.as_deref());
    Ok(refresh::build_refresh_cookie(
        &raw_token,
        auth_config.refresh_token_expiry_secs,
        is_secure,
    ))
}

/// Build an HTTP response with the access token JSON body + Set-Cookie header
/// for the refresh token.
fn auth_response_with_cookie(
    json_body: AuthTokenResponse,
    cookie: axum::http::HeaderValue,
) -> Response {
    (
        [(SET_COOKIE, cookie)],
        Json(json_body),
    )
        .into_response()
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
/// 5. Return JWT access token in body + refresh token in HttpOnly cookie
///
/// Security: error messages never reveal whether the email exists or not.
pub async fn password_login(
    State(state): State<OrchestratorState>,
    Json(req): Json<LoginRequest>,
) -> Result<Response, AppError> {
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
    let invalid_credentials = || AppError::Unauthorized("Invalid email or password".to_string());

    // 1. Check root account first (in-memory, no DB)
    if let Some(ref root) = auth_config.root_account {
        if req.email == root.email {
            let password_ok = bcrypt::verify(&req.password, &root.password_hash).unwrap_or(false);
            if password_ok {
                // Root user gets a deterministic UUID based on email
                let root_user_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes());
                let token = encode_jwt(
                    root_user_id,
                    &root.email,
                    &root.name,
                    &auth_config.jwt_secret,
                    auth_config.access_token_expiry_secs,
                )
                .map_err(AppError::Internal)?;

                // 5. Generate refresh token + Set-Cookie
                let cookie = create_refresh_token_and_cookie(&state, auth_config, root_user_id).await?;

                return Ok(auth_response_with_cookie(
                    AuthTokenResponse {
                        token,
                        user: UserResponse {
                            id: root_user_id,
                            email: root.email.clone(),
                            name: root.name.clone(),
                            picture_url: None,
                            is_root: true,
                        },
                    },
                    cookie,
                ));
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

    let password_ok = bcrypt::verify(&req.password, password_hash).unwrap_or(false);
    if !password_ok {
        return Err(invalid_credentials());
    }

    // 4. Generate JWT
    let token = encode_jwt(
        user.id,
        &user.email,
        &user.name,
        &auth_config.jwt_secret,
        auth_config.access_token_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    // 5. Generate refresh token + Set-Cookie
    let cookie = create_refresh_token_and_cookie(&state, auth_config, user.id).await?;

    Ok(auth_response_with_cookie(
        AuthTokenResponse {
            token,
            user: UserResponse::from(user),
        },
        cookie,
    ))
}

/// POST /auth/register — Create a new password-authenticated account.
///
/// Only available when `allow_registration` is true in auth config.
/// Creates a user in Neo4j with bcrypt-hashed password and returns
/// JWT + user info (auto-login after registration).
pub async fn register(
    State(state): State<OrchestratorState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // 1. Check registration is enabled
    if !auth_config.allow_registration {
        return Err(AppError::Forbidden("Registration is disabled".to_string()));
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
        auth_config.access_token_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    // 7. Generate refresh token + Set-Cookie
    let cookie = create_refresh_token_and_cookie(&state, auth_config, user.id).await?;

    Ok(auth_response_with_cookie(
        AuthTokenResponse {
            token,
            user: UserResponse::from(user),
        },
        cookie,
    ))
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
    if !email.contains('@') || !email.split('@').nth(1).is_some_and(|d| d.contains('.')) {
        return Err(AppError::BadRequest("Invalid email format".to_string()));
    }

    // Password minimum length
    if req.password.len() < 8 {
        return Err(AppError::BadRequest(
            "Password must be at least 8 characters".to_string(),
        ));
    }

    // Check email restrictions (domain + individual whitelist)
    if !auth_config.is_email_allowed(&email) {
        return Err(AppError::Forbidden(
            "Email not allowed by server policy".to_string(),
        ));
    }

    Ok(())
}

/// GET /auth/google — Returns the Google OAuth authorization URL.
///
/// Accepts an optional `?origin=` query parameter. When provided, the origin is
/// validated against the allowed origins whitelist and used to build a dynamic
/// `redirect_uri` (e.g. `https://ffs.dev/auth/callback` for web access).
/// When omitted, falls back to the static `redirect_uri` from config.
pub async fn google_login(
    State(state): State<OrchestratorState>,
    AxumQuery(query): AxumQuery<OAuthLoginQuery>,
) -> Result<Json<AuthUrlResponse>, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    let client = GoogleOAuthClient::new(auth_config);

    let auth_url = match state.validate_origin(query.origin.as_deref())? {
        Some(origin) => {
            let redirect_uri = format!("{}/auth/callback", origin);
            client.auth_url_with_redirect(&redirect_uri)
        }
        None => client.auth_url(),
    };

    Ok(Json(AuthUrlResponse { auth_url }))
}

/// POST /auth/google/callback — Exchange authorization code for JWT + user.
///
/// 1. Exchanges the Google auth code for user info (using dynamic redirect_uri if origin provided)
/// 2. Upserts the user in Neo4j (creates on first login, updates on subsequent)
/// 3. Returns a JWT token + user info
pub async fn google_callback(
    State(state): State<OrchestratorState>,
    Json(req): Json<GoogleCallbackRequest>,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // 1. Exchange code for Google user info (redirect_uri must match the one used in auth URL)
    let client = GoogleOAuthClient::new(auth_config);
    let google_user = match state.validate_origin(req.origin.as_deref())? {
        Some(origin) => {
            let redirect_uri = format!("{}/auth/callback", origin);
            client
                .exchange_code_with_redirect(&req.code, &redirect_uri)
                .await
        }
        None => client.exchange_code(&req.code).await,
    }
    .map_err(|e| AppError::BadRequest(format!("OAuth code exchange failed: {}", e)))?;

    // 2. Check email restrictions (domain + individual whitelist)
    if !auth_config.is_email_allowed(&google_user.email) {
        return Err(AppError::Forbidden(
            "Email not allowed by server policy".to_string(),
        ));
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
        auth_config.access_token_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    // 5. Generate refresh token + Set-Cookie
    let cookie = create_refresh_token_and_cookie(&state, auth_config, user.id).await?;

    Ok(auth_response_with_cookie(
        AuthTokenResponse {
            token,
            user: UserResponse::from(user),
        },
        cookie,
    ))
}

/// GET /auth/oidc — Returns the OIDC authorization URL.
///
/// Uses the generic OIDC client built from `effective_oidc()` config
/// or legacy Google fields. The frontend redirects the user to this URL.
///
/// Accepts an optional `?origin=` query parameter for dynamic redirect_uri
/// construction (see `google_login` for details).
pub async fn oidc_login(
    State(state): State<OrchestratorState>,
    AxumQuery(query): AxumQuery<OAuthLoginQuery>,
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

    let client = OidcClient::from_auth_config_sync(auth_config).map_err(AppError::Internal)?;

    let auth_url = match state.validate_origin(query.origin.as_deref())? {
        Some(origin) => {
            let redirect_uri = format!("{}/auth/callback", origin);
            client.auth_url_with_redirect(&redirect_uri)
        }
        None => client.auth_url(),
    };

    Ok(Json(AuthUrlResponse { auth_url }))
}

/// POST /auth/oidc/callback — Exchange OIDC authorization code for JWT + user.
///
/// 1. Exchanges the auth code via the generic OIDC client (using dynamic redirect_uri if origin provided)
/// 2. Upserts the user in Neo4j with auth_provider="oidc" + external_id=sub
/// 3. Returns a JWT token + user info
pub async fn oidc_callback(
    State(state): State<OrchestratorState>,
    Json(req): Json<OidcCallbackRequest>,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    if !auth_config.has_oidc() {
        return Err(AppError::Forbidden(
            "OIDC authentication is not configured".to_string(),
        ));
    }

    // 1. Build OIDC client and exchange code (redirect_uri must match the one used in auth URL)
    let client = OidcClient::from_auth_config_sync(auth_config).map_err(AppError::Internal)?;

    let oidc_user = match state.validate_origin(req.origin.as_deref())? {
        Some(origin) => {
            let redirect_uri = format!("{}/auth/callback", origin);
            client
                .exchange_code_with_redirect(&req.code, &redirect_uri)
                .await
        }
        None => client.exchange_code(&req.code).await,
    }
    .map_err(|e| AppError::BadRequest(format!("OIDC code exchange failed: {}", e)))?;

    // 2. Check email restrictions (domain + individual whitelist)
    if !auth_config.is_email_allowed(&oidc_user.email) {
        return Err(AppError::Forbidden(
            "Email not allowed by server policy".to_string(),
        ));
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
        auth_config.access_token_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    // 5. Generate refresh token + Set-Cookie
    let cookie = create_refresh_token_and_cookie(&state, auth_config, user.id).await?;

    Ok(auth_response_with_cookie(
        AuthTokenResponse {
            token,
            user: UserResponse::from(user),
        },
        cookie,
    ))
}

/// GET /auth/me — Returns the authenticated user's info.
///
/// Requires a valid JWT token (via `require_auth` middleware + `AuthUser` extractor).
/// Falls back to JWT claims when the user isn't found in Neo4j (e.g. root account).
/// Sets `is_root: true` when the user ID matches the root account's deterministic UUID.
pub async fn get_me(
    State(state): State<OrchestratorState>,
    user: AuthUser,
) -> Result<Json<UserResponse>, AppError> {
    // Check if this user is the root account (deterministic UUID from email)
    let is_root = state
        .auth_config
        .as_ref()
        .and_then(|c| c.root_account.as_ref())
        .is_some_and(|root| {
            Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes()) == user.user_id
        });

    // Try to fetch full user from Neo4j (may have updated picture, etc.)
    match state
        .orchestrator
        .neo4j()
        .get_user_by_id(user.user_id)
        .await?
    {
        Some(user_node) => {
            let mut resp = UserResponse::from(user_node);
            resp.is_root = is_root;
            Ok(Json(resp))
        }
        // Root account or users not yet in Neo4j — use JWT claims directly
        None => Ok(Json(UserResponse {
            id: user.user_id,
            email: user.email,
            name: user.name,
            picture_url: None,
            is_root,
        })),
    }
}

/// POST /auth/refresh — Issue a new access JWT from a valid refresh cookie.
///
/// Reads the `refresh_token` cookie (HttpOnly, set by login/register),
/// validates it against the database (not expired, not revoked), then:
/// 1. Revokes the old refresh token (rotation — single use)
/// 2. Creates a new refresh token and emits a new Set-Cookie
/// 3. Returns a fresh access JWT in the body
///
/// This does NOT require a Bearer header — the refresh cookie is sufficient.
/// This allows refreshing even after the access token has expired.
pub async fn refresh_token(
    State(state): State<OrchestratorState>,
    headers: axum::http::HeaderMap,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // 1. Extract refresh token from Cookie header
    let cookie_header = headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("No refresh token cookie".to_string()))?;

    let raw_token = refresh::extract_refresh_token_from_cookie(cookie_header)
        .ok_or_else(|| AppError::Unauthorized("No refresh_token in cookie".to_string()))?;

    // 2. Validate refresh token in DB (checks expiry + revoked)
    let token_hash = refresh::hash_token(&raw_token);
    let token_node = state
        .orchestrator
        .neo4j()
        .validate_refresh_token(&token_hash)
        .await?
        .ok_or_else(|| {
            AppError::Unauthorized("Invalid or expired refresh token".to_string())
        })?;

    // 3. Revoke old refresh token (rotation — each token is single-use)
    state
        .orchestrator
        .neo4j()
        .revoke_refresh_token(&token_hash)
        .await?;

    // 4. Look up user info for JWT claims
    let user = state
        .orchestrator
        .neo4j()
        .get_user_by_id(token_node.user_id)
        .await?;

    // For root accounts (not in Neo4j), fall back to auth config
    let (user_id, email, name) = match user {
        Some(u) => (u.id, u.email, u.name),
        None => {
            // Check if this is the root account
            if let Some(ref root) = auth_config.root_account {
                let root_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes());
                if root_id == token_node.user_id {
                    (root_id, root.email.clone(), root.name.clone())
                } else {
                    return Err(AppError::Unauthorized("User not found".to_string()));
                }
            } else {
                return Err(AppError::Unauthorized("User not found".to_string()));
            }
        }
    };

    // 5. Generate new access JWT
    let access_token = encode_jwt(
        user_id,
        &email,
        &name,
        &auth_config.jwt_secret,
        auth_config.access_token_expiry_secs,
    )
    .map_err(AppError::Internal)?;

    // 6. Generate new refresh token + Set-Cookie (rotation)
    let cookie = create_refresh_token_and_cookie(&state, auth_config, user_id).await?;

    Ok((
        [(SET_COOKIE, cookie)],
        Json(RefreshTokenResponse { token: access_token }),
    )
        .into_response())
}

/// POST /auth/logout — Revoke the refresh token and clear the cookie.
///
/// Reads the `refresh_token` cookie, revokes the token hash in the database,
/// and returns a `Set-Cookie` header that clears the cookie (Max-Age=0).
///
/// This is a public route (no Bearer required) — the user may have an expired
/// access token at logout time. The cookie itself is the credential.
///
/// If no cookie is present, the endpoint returns 200 (idempotent — already logged out).
pub async fn logout(
    State(state): State<OrchestratorState>,
    headers: axum::http::HeaderMap,
) -> Result<Response, AppError> {
    // If auth is not configured, logout is a no-op
    let _auth_config = match state.auth_config.as_ref() {
        Some(c) => c,
        None => {
            return Ok(axum::http::StatusCode::NO_CONTENT.into_response());
        }
    };

    // Extract refresh token from cookie (if present)
    let raw_token = headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(refresh::extract_refresh_token_from_cookie);

    // Revoke the token in DB (if we have one)
    if let Some(token) = raw_token {
        let token_hash = refresh::hash_token(&token);
        // Best-effort revocation — don't fail if token is already revoked or not found
        let _ = state
            .orchestrator
            .neo4j()
            .revoke_refresh_token(&token_hash)
            .await;
    }

    // Clear the cookie regardless (idempotent)
    let is_secure = refresh::should_set_secure(state.public_url.as_deref());
    let clear_cookie = refresh::build_clear_cookie(is_secure);

    Ok(([(SET_COOKIE, clear_cookie)], axum::http::StatusCode::NO_CONTENT).into_response())
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
            access_token_expiry_secs: 900,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
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
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            EventBus::default(),
        )));
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
            nats_emitter: None,
            auth_config,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
        })
    }

    /// Build a test router with auth routes (some protected, some public)
    async fn test_auth_app(auth_config: Option<AuthConfig>) -> Router {
        let state = make_server_state(auth_config).await;

        // Public auth routes (no middleware — refresh/logout read cookie, not Bearer)
        let public = Router::new()
            .route("/auth/providers", get(get_auth_providers))
            .route("/auth/login", post(password_login))
            .route("/auth/register", post(register))
            .route("/auth/refresh", post(refresh_token))
            .route("/auth/logout", post(logout))
            .route("/auth/google", get(google_login));

        // Protected auth routes (require Bearer JWT)
        let protected = Router::new()
            .route("/auth/me", get(get_me))
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
            .route("/auth/refresh", post(refresh_token))
            .route("/auth/logout", post(logout))
            .route("/auth/google", get(google_login));

        let protected = Router::new()
            .route("/auth/me", get(get_me))
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
    async fn test_refresh_with_cookie_returns_new_token() {
        let state = make_server_state(Some(test_auth_config())).await;

        // Create user in mock store
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

        // Create a refresh token in DB (simulating what login would do)
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = Utc::now() + chrono::Duration::days(7);
        state
            .orchestrator
            .neo4j()
            .create_refresh_token(user_id, &token_hash, expires_at)
            .await
            .unwrap();

        // Build app (refresh is public — no Bearer needed)
        let app = Router::new()
            .route("/auth/refresh", post(refresh_token))
            .with_state(state);

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify Set-Cookie header is present (new refresh token after rotation)
        let set_cookie = resp
            .headers()
            .get("set-cookie")
            .expect("Set-Cookie header must be present after refresh");
        let cookie_str = set_cookie.to_str().unwrap();
        assert!(cookie_str.contains("refresh_token="), "Cookie must contain refresh_token");
        assert!(cookie_str.contains("HttpOnly"), "Cookie must be HttpOnly");
        assert!(cookie_str.contains("Path=/"), "Cookie must have Path=/");

        // Verify the body contains a new access JWT
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let new_access_token = json["token"].as_str().unwrap();
        assert!(!new_access_token.is_empty());

        // Decode to verify claims
        let claims = crate::auth::jwt::decode_jwt(new_access_token, TEST_SECRET).unwrap();
        assert_eq!(claims.sub, user_id.to_string());
        assert_eq!(claims.email, "bob@ffs.holdings");
    }

    #[tokio::test]
    async fn test_refresh_without_cookie_returns_401() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_refresh_with_invalid_cookie_returns_401() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", "refresh_token=invalid-token-value")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_refresh_revoked_token_returns_401() {
        let state = make_server_state(Some(test_auth_config())).await;

        let user_id = Uuid::new_v4();
        let now = Utc::now();
        let user_node = UserNode {
            id: user_id,
            email: "carol@ffs.holdings".to_string(),
            name: "Carol".to_string(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Oidc,
            external_id: Some("google-789".to_string()),
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

        // Create and immediately revoke a refresh token
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = Utc::now() + chrono::Duration::days(7);
        state
            .orchestrator
            .neo4j()
            .create_refresh_token(user_id, &token_hash, expires_at)
            .await
            .unwrap();
        state
            .orchestrator
            .neo4j()
            .revoke_refresh_token(&token_hash)
            .await
            .unwrap();

        let app = Router::new()
            .route("/auth/refresh", post(refresh_token))
            .with_state(state);

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_refresh_rotation_old_token_invalidated() {
        let state = make_server_state(Some(test_auth_config())).await;

        let user_id = Uuid::new_v4();
        let now = Utc::now();
        let user_node = UserNode {
            id: user_id,
            email: "dave@ffs.holdings".to_string(),
            name: "Dave".to_string(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Oidc,
            external_id: Some("google-101".to_string()),
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

        // Create a refresh token
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = Utc::now() + chrono::Duration::days(7);
        state
            .orchestrator
            .neo4j()
            .create_refresh_token(user_id, &token_hash, expires_at)
            .await
            .unwrap();

        // First refresh — should succeed
        let app = Router::new()
            .route("/auth/refresh", post(refresh_token))
            .with_state(state.clone());

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second refresh with same token — should fail (already revoked by rotation)
        let app2 = Router::new()
            .route("/auth/refresh", post(refresh_token))
            .with_state(state);

        let req2 = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp2 = app2.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::UNAUTHORIZED);
    }

    // ================================================================
    // Logout tests
    // ================================================================

    #[tokio::test]
    async fn test_logout_revokes_token_and_clears_cookie() {
        let state = make_server_state(Some(test_auth_config())).await;

        let user_id = Uuid::new_v4();
        let now = Utc::now();
        let user_node = UserNode {
            id: user_id,
            email: "eve@ffs.holdings".to_string(),
            name: "Eve".to_string(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Password,
            external_id: None,
            password_hash: Some("hash".to_string()),
            created_at: now,
            last_login_at: now,
        };
        state.orchestrator.neo4j().upsert_user(&user_node).await.unwrap();

        // Create a refresh token
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = Utc::now() + chrono::Duration::days(7);
        state
            .orchestrator
            .neo4j()
            .create_refresh_token(user_id, &token_hash, expires_at)
            .await
            .unwrap();

        let app = Router::new()
            .route("/auth/logout", post(logout))
            .route("/auth/refresh", post(refresh_token))
            .with_state(state.clone());

        // Logout
        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/logout")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Verify Set-Cookie clears the cookie
        let set_cookie = resp
            .headers()
            .get("set-cookie")
            .expect("Logout must return Set-Cookie header");
        let cookie_str = set_cookie.to_str().unwrap();
        assert!(cookie_str.contains("refresh_token=;"), "Cookie value must be cleared");
        assert!(cookie_str.contains("Max-Age=0"), "Cookie must expire immediately");
        assert!(cookie_str.contains("HttpOnly"), "Cookie must be HttpOnly");
        assert!(cookie_str.contains("Path=/"), "Cookie must have Path=/");

        // Verify the token is now revoked — refresh should fail
        let req2 = HttpRequest::builder()
            .method("POST")
            .uri("/auth/refresh")
            .header("cookie", format!("refresh_token={}", raw_token))
            .body(Body::empty())
            .unwrap();

        let resp2 = app.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_logout_without_cookie_is_idempotent() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/logout")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Should still clear the cookie
        let set_cookie = resp.headers().get("set-cookie");
        assert!(set_cookie.is_some(), "Should still send Set-Cookie to clear");
    }

    #[tokio::test]
    async fn test_logout_with_invalid_token_still_clears_cookie() {
        let app = test_auth_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/auth/logout")
            .header("cookie", "refresh_token=nonexistent_token")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Should still clear the cookie (idempotent)
        let set_cookie = resp
            .headers()
            .get("set-cookie")
            .expect("Should send Set-Cookie even for invalid token");
        let cookie_str = set_cookie.to_str().unwrap();
        assert!(cookie_str.contains("Max-Age=0"));
    }

    // ================================================================
    // Password login tests
    // ================================================================

    fn auth_config_with_root() -> AuthConfig {
        let password_hash = bcrypt::hash("rootpass123", 4).unwrap(); // cost 4 for fast tests
        AuthConfig {
            jwt_secret: TEST_SECRET.to_string(),
            access_token_expiry_secs: 900,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
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

        // Verify Set-Cookie header is present with correct flags
        let set_cookie = resp
            .headers()
            .get("set-cookie")
            .expect("Login must return Set-Cookie header");
        let cookie_str = set_cookie.to_str().unwrap();
        assert!(cookie_str.contains("refresh_token="), "Cookie must contain refresh_token");
        assert!(cookie_str.contains("HttpOnly"), "Cookie must be HttpOnly");
        assert!(cookie_str.contains("SameSite=Lax"), "Cookie must be SameSite=Lax");
        assert!(cookie_str.contains("Path=/"), "Cookie must have Path=/");

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["token"].as_str().is_some());
        assert_eq!(json["user"]["email"], "admin@ffs.holdings");
        assert_eq!(json["user"]["name"], "Admin");

        // Verify the access token is valid
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

        let types: Vec<&str> = providers
            .iter()
            .map(|p| p["type"].as_str().unwrap())
            .collect();
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
            .body(register_body(
                "newuser@ffs.holdings",
                "securepass123",
                "New User",
            ))
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
            .body(register_body(
                "newuser@ffs.holdings",
                "securepass123",
                "New User",
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_register_duplicate_email_returns_409() {
        let (app, state) = test_auth_app_with_state(Some(auth_config_with_registration())).await;

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
            .body(register_body(
                "existing@ffs.holdings",
                "newpass123",
                "Another",
            ))
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
            access_token_expiry_secs: 900,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
            frontend_url: None,
            allow_registration: false,
            root_account: None,
            oidc: Some(crate::OidcConfig {
                provider_key: None,
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
