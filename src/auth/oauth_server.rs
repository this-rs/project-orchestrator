//! OAuth 2.1 Authorization Server — for Claude.ai remote MCP connectors.
//!
//! Claude.ai (web, desktop AND mobile apps) connects to remote MCP servers
//! through custom connectors that REQUIRE the OAuth 2.1 flow: metadata
//! discovery, Dynamic Client Registration, and the authorization-code +
//! PKCE (S256) exchange. This module makes PO its own Authorization Server:
//!
//! - `GET /.well-known/oauth-authorization-server` — RFC 8414 metadata
//! - `GET /.well-known/oauth-protected-resource` — RFC 9728 metadata
//! - `POST /oauth/register` — RFC 7591 Dynamic Client Registration
//! - `GET /oauth/authorize` — authorization-code grant with PKCE S256
//! - `POST /oauth/token` — code exchange → long-lived scoped MCP token
//!
//! ## User authentication
//! `/oauth/authorize` identifies the user via the existing HttpOnly
//! `refresh_token` cookie (same mechanism as the WS upgrade auth). A browser
//! with an active PO session authorizes silently; otherwise the user is
//! redirected to the SPA `/login?next=<authorize-url>` to sign in first.
//!
//! ## Stateless clients
//! Registered clients are PUBLIC clients (no secret) whose only state is
//! `client_name` + `redirect_uris`. That state is encoded INTO the
//! `client_id` itself as an HS256-signed JWT (existing `jwt_secret`):
//! zero storage, restart-proof, tamper-proof (plan decision 1df0be0d).
//!
//! ## Issued tokens
//! `/oauth/token` issues the same revocable scoped MCP tokens as
//! `POST /auth/mcp-tokens` (label `OAuth: {client_name}`) — inventory and
//! revocation work identically for OAuth-issued and manually-issued tokens.

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::encode_mcp_token;
use crate::auth::refresh;
use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Redirect, Response},
    Form, Json,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use uuid::Uuid;

/// Authorization codes are single-use and short-lived.
const AUTH_CODE_TTL: Duration = Duration::from_secs(120);

/// Lifetime of OAuth-issued MCP access tokens (30 days). No refresh_token
/// grant yet — when the token expires the connector re-runs the (silent,
/// cookie-backed) authorization flow.
const OAUTH_ACCESS_TOKEN_EXPIRY_SECS: u64 = 30 * 86_400;

/// Client-id JWTs are practically non-expiring (10 years).
const CLIENT_ID_EXPIRY_SECS: i64 = 10 * 365 * 86_400;

/// Scope granted to OAuth-issued tokens.
const OAUTH_TOKEN_SCOPE: &str = "mcp:read mcp:write";

// ============================================================================
// Helpers — base64url + PKCE
// ============================================================================

/// Base64url (RFC 4648 §5) without padding — encode-only, no dependency.
fn base64url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[n as usize & 63] as char);
        }
    }
    out
}

/// PKCE S256 check: `BASE64URL(SHA256(code_verifier)) == code_challenge`.
fn pkce_matches(code_verifier: &str, code_challenge: &str) -> bool {
    let digest = Sha256::digest(code_verifier.as_bytes());
    base64url_encode(&digest) == code_challenge
}

/// Percent-encode a query-string component (conservative: everything except
/// RFC 3986 unreserved characters).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// The canonical external base URL of this server (no trailing slash).
fn issuer(state: &OrchestratorState) -> String {
    state
        .public_url
        .clone()
        .unwrap_or_else(|| format!("http://localhost:{}", state.server_port))
        .trim_end_matches('/')
        .to_string()
}

// ============================================================================
// Stateless client registration (client_id = signed JWT)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct ClientIdClaims {
    /// Fixed subject so these JWTs can never be confused with user tokens.
    sub: String,
    client_name: String,
    redirect_uris: Vec<String>,
    iat: i64,
    exp: i64,
}

const CLIENT_ID_SUBJECT: &str = "oauth-client-registration";

fn mint_client_id(client_name: &str, redirect_uris: &[String], secret: &str) -> Option<String> {
    let now = chrono::Utc::now().timestamp();
    let claims = ClientIdClaims {
        sub: CLIENT_ID_SUBJECT.to_string(),
        client_name: client_name.to_string(),
        redirect_uris: redirect_uris.to_vec(),
        iat: now,
        exp: now + CLIENT_ID_EXPIRY_SECS,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .ok()
}

fn validate_client_id(client_id: &str, secret: &str) -> Option<ClientIdClaims> {
    let data = decode::<ClientIdClaims>(
        client_id,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .ok()?;
    (data.claims.sub == CLIENT_ID_SUBJECT).then_some(data.claims)
}

/// A redirect_uri is acceptable when it is HTTPS, or plain HTTP on
/// localhost/127.0.0.1 (native-app loopback, per OAuth 2.1).
fn redirect_uri_allowed(uri: &str) -> bool {
    if uri.starts_with("https://") {
        return true;
    }
    if let Some(rest) = uri.strip_prefix("http://") {
        let host = rest.split(['/', ':', '?', '#']).next().unwrap_or("");
        return host == "localhost" || host == "127.0.0.1";
    }
    false
}

// ============================================================================
// Auth-code registry (in-memory, single-use, 120s TTL)
// ============================================================================

struct AuthCodeEntry {
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    user_id: Uuid,
    email: String,
    name: String,
    created: Instant,
}

fn auth_codes() -> &'static RwLock<HashMap<String, AuthCodeEntry>> {
    static CODES: OnceLock<RwLock<HashMap<String, AuthCodeEntry>>> = OnceLock::new();
    CODES.get_or_init(|| RwLock::new(HashMap::new()))
}

fn store_auth_code(entry: AuthCodeEntry) -> String {
    let code = Uuid::new_v4().to_string();
    let mut map = auth_codes().write().expect("auth code lock poisoned");
    map.retain(|_, e| e.created.elapsed() < AUTH_CODE_TTL);
    map.insert(code.clone(), entry);
    code
}

/// Take (and consume) an auth code — single use by construction.
fn take_auth_code(code: &str) -> Option<AuthCodeEntry> {
    let mut map = auth_codes().write().expect("auth code lock poisoned");
    let entry = map.remove(code)?;
    (entry.created.elapsed() < AUTH_CODE_TTL).then_some(entry)
}

// ============================================================================
// Metadata endpoints
// ============================================================================

/// GET /.well-known/oauth-authorization-server (RFC 8414 — public).
pub async fn authorization_server_metadata(State(state): State<OrchestratorState>) -> Response {
    let base = issuer(&state);
    Json(json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/oauth/authorize"),
        "token_endpoint": format!("{base}/oauth/token"),
        "registration_endpoint": format!("{base}/oauth/register"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp:read", "mcp:write"],
    }))
    .into_response()
}

/// GET /.well-known/oauth-protected-resource (RFC 9728 — public).
pub async fn protected_resource_metadata(State(state): State<OrchestratorState>) -> Response {
    let base = issuer(&state);
    Json(json!({
        "resource": format!("{base}/mcp"),
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp:read", "mcp:write"],
    }))
    .into_response()
}

// ============================================================================
// Dynamic Client Registration (RFC 7591)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub redirect_uris: Vec<String>,
}

/// POST /oauth/register — public (RFC 7591 open registration for public
/// clients). The returned client_id is a signed JWT embedding the
/// registration (see module docs).
pub async fn register_client(
    State(state): State<OrchestratorState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    if req.redirect_uris.is_empty() {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_client_metadata",
            "redirect_uris is required",
        ));
    }
    if let Some(bad) = req.redirect_uris.iter().find(|u| !redirect_uri_allowed(u)) {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_redirect_uri",
            &format!("redirect_uri not allowed: {bad}"),
        ));
    }

    let client_name = req
        .client_name
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("Unnamed client")
        .trim()
        .to_string();

    let client_id = mint_client_id(&client_name, &req.redirect_uris, &auth_config.jwt_secret)
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("failed to mint client_id")))?;

    info!(client = %client_name, uris = ?req.redirect_uris, "OAuth client registered (stateless)");
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "client_id": client_id,
            "client_name": client_name,
            "redirect_uris": req.redirect_uris,
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "client_id_issued_at": chrono::Utc::now().timestamp(),
        })),
    )
        .into_response())
}

// ============================================================================
// Authorize (authorization-code + PKCE S256)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AuthorizeQuery {
    #[serde(default)]
    pub response_type: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub redirect_uri: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub code_challenge: Option<String>,
    #[serde(default)]
    pub code_challenge_method: Option<String>,
    // scope / resource accepted but not restricted per-request yet.
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub resource: Option<String>,
}

/// GET /oauth/authorize — public route; the USER is authenticated via the
/// HttpOnly `refresh_token` cookie (existing PO session). No session →
/// redirect to the SPA login with `next=` back here.
pub async fn authorize(
    State(state): State<OrchestratorState>,
    axum::extract::RawQuery(raw_query): axum::extract::RawQuery,
    Query(q): Query<AuthorizeQuery>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    // --- Validate client + redirect_uri FIRST (never redirect to an
    //     unvalidated URI — errors here return 400 directly).
    let Some(client_id) = q.client_id.as_deref() else {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "client_id is required",
        ));
    };
    let Some(client) = validate_client_id(client_id, &auth_config.jwt_secret) else {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_client",
            "unknown client_id",
        ));
    };
    let Some(redirect_uri) = q.redirect_uri.as_deref() else {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri is required",
        ));
    };
    if !client.redirect_uris.iter().any(|u| u == redirect_uri) {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri not registered for this client",
        ));
    }

    // --- Remaining validation errors redirect back to the client.
    let state_param = q.state.as_deref().unwrap_or("");
    let err_redirect = |err: &str, desc: &str| -> Response {
        let mut loc = format!(
            "{redirect_uri}{}error={}&error_description={}",
            if redirect_uri.contains('?') { "&" } else { "?" },
            urlencode(err),
            urlencode(desc),
        );
        if !state_param.is_empty() {
            loc.push_str(&format!("&state={}", urlencode(state_param)));
        }
        Redirect::to(&loc).into_response()
    };

    if q.response_type.as_deref() != Some("code") {
        return Ok(err_redirect(
            "unsupported_response_type",
            "only response_type=code is supported",
        ));
    }
    let Some(code_challenge) = q.code_challenge.as_deref().filter(|c| !c.is_empty()) else {
        return Ok(err_redirect(
            "invalid_request",
            "PKCE code_challenge is required",
        ));
    };
    // OAuth 2.1: S256 only ("plain" is forbidden). Absent method defaults to
    // "plain" per RFC 7636, so an explicit S256 is required.
    if q.code_challenge_method.as_deref() != Some("S256") {
        return Ok(err_redirect(
            "invalid_request",
            "code_challenge_method must be S256",
        ));
    }

    // --- Identify the user via the refresh cookie (existing PO session).
    let user = user_from_refresh_cookie(&state, auth_config, &headers).await;
    let Some((user_id, email, name)) = user else {
        // No active session → send the browser to the SPA login, then back.
        let next = format!(
            "{}/oauth/authorize?{}",
            issuer(&state),
            raw_query.unwrap_or_default()
        );
        let login = format!("{}/login?next={}", issuer(&state), urlencode(&next));
        return Ok(Redirect::to(&login).into_response());
    };

    // --- Issue the single-use code and bounce back to the client.
    let code = store_auth_code(AuthCodeEntry {
        client_id: client_id.to_string(),
        redirect_uri: redirect_uri.to_string(),
        code_challenge: code_challenge.to_string(),
        user_id,
        email: email.clone(),
        name,
        created: Instant::now(),
    });
    info!(user = %email, client = %client.client_name, "OAuth authorization code issued");

    let mut loc = format!(
        "{redirect_uri}{}code={}",
        if redirect_uri.contains('?') { "&" } else { "?" },
        urlencode(&code),
    );
    if !state_param.is_empty() {
        loc.push_str(&format!("&state={}", urlencode(state_param)));
    }
    Ok(Redirect::to(&loc).into_response())
}

/// Resolve the current user from the HttpOnly refresh cookie (same
/// mechanism as the WS upgrade auth in `ws_auth.rs`).
async fn user_from_refresh_cookie(
    state: &OrchestratorState,
    auth_config: &crate::AuthConfig,
    headers: &HeaderMap,
) -> Option<(Uuid, String, String)> {
    let cookie_header = headers.get(header::COOKIE)?.to_str().ok()?;
    let raw_token = refresh::extract_refresh_token_from_cookie(cookie_header)?;
    let token_hash = refresh::hash_token(&raw_token);
    let token_node = state
        .orchestrator
        .neo4j()
        .validate_refresh_token(&token_hash)
        .await
        .ok()??;

    match state
        .orchestrator
        .neo4j()
        .get_user_by_id(token_node.user_id)
        .await
    {
        Ok(Some(user)) => Some((user.id, user.email, user.name)),
        Ok(None) => {
            // Root account fallback — root users are not stored in Neo4j.
            let root = auth_config.root_account.as_ref()?;
            let root_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes());
            (root_id == token_node.user_id)
                .then(|| (root_id, root.email.clone(), root.name.clone()))
        }
        Err(_) => None,
    }
}

// ============================================================================
// Token endpoint
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    #[serde(default)]
    pub grant_type: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub redirect_uri: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub code_verifier: Option<String>,
}

/// POST /oauth/token — exchange an authorization code + PKCE verifier for a
/// revocable scoped MCP access token. Form-encoded per OAuth.
pub async fn token(
    State(state): State<OrchestratorState>,
    Form(req): Form<TokenRequest>,
) -> Result<Response, AppError> {
    let auth_config = state
        .auth_config
        .as_ref()
        .ok_or_else(|| AppError::Forbidden("Authentication not configured".to_string()))?;

    if req.grant_type.as_deref() != Some("authorization_code") {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            "only authorization_code is supported",
        ));
    }
    let (Some(code), Some(verifier)) = (req.code.as_deref(), req.code_verifier.as_deref()) else {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "code and code_verifier are required",
        ));
    };

    // Single-use: the code is consumed on lookup, success or not afterwards.
    let Some(entry) = take_auth_code(code) else {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "unknown, expired or already-used code",
        ));
    };
    if req.client_id.as_deref() != Some(entry.client_id.as_str()) {
        warn!("OAuth token exchange: client_id mismatch");
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "client_id mismatch",
        ));
    }
    if req.redirect_uri.as_deref() != Some(entry.redirect_uri.as_str()) {
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "redirect_uri mismatch",
        ));
    }
    if !pkce_matches(verifier, &entry.code_challenge) {
        warn!(user = %entry.email, "OAuth token exchange: PKCE verification failed");
        return Ok(oauth_error_json(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "PKCE verification failed",
        ));
    }

    // Issue a standard revocable MCP token (shared inventory/revocation with
    // POST /auth/mcp-tokens).
    let client_name = validate_client_id(&entry.client_id, &auth_config.jwt_secret)
        .map(|c| c.client_name)
        .unwrap_or_else(|| "OAuth client".to_string());
    let (access_token, jti) = encode_mcp_token(
        entry.user_id,
        &entry.email,
        &entry.name,
        OAUTH_TOKEN_SCOPE,
        &auth_config.jwt_secret,
        OAUTH_ACCESS_TOKEN_EXPIRY_SECS,
    )
    .map_err(AppError::Internal)?;
    let expires_at =
        chrono::Utc::now() + chrono::Duration::seconds(OAUTH_ACCESS_TOKEN_EXPIRY_SECS as i64);
    state
        .orchestrator
        .neo4j()
        .create_mcp_token(
            entry.user_id,
            &jti,
            &format!("OAuth: {client_name}"),
            OAUTH_TOKEN_SCOPE,
            expires_at,
        )
        .await?;

    info!(user = %entry.email, client = %client_name, jti = %jti, "OAuth MCP token issued");
    Ok(Json(json!({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": OAUTH_ACCESS_TOKEN_EXPIRY_SECS,
        "scope": OAUTH_TOKEN_SCOPE,
    }))
    .into_response())
}

/// OAuth-style JSON error body.
fn oauth_error_json(status: StatusCode, error: &str, description: &str) -> Response {
    (
        status,
        Json(json!({ "error": error, "error_description": description })),
    )
        .into_response()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "test-secret-key-minimum-32-chars!!";

    #[test]
    fn test_base64url_rfc_vectors() {
        // RFC 4648 §10 vectors, unpadded.
        assert_eq!(base64url_encode(b""), "");
        assert_eq!(base64url_encode(b"f"), "Zg");
        assert_eq!(base64url_encode(b"fo"), "Zm8");
        assert_eq!(base64url_encode(b"foo"), "Zm9v");
        assert_eq!(base64url_encode(b"foob"), "Zm9vYg");
        assert_eq!(base64url_encode(b"fooba"), "Zm9vYmE");
        assert_eq!(base64url_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_pkce_s256_rfc_vector() {
        // RFC 7636 Appendix B reference values.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
        assert!(pkce_matches(verifier, challenge));
        assert!(!pkce_matches("wrong-verifier", challenge));
    }

    #[test]
    fn test_client_id_roundtrip() {
        let uris = vec!["https://claude.ai/api/mcp/auth_callback".to_string()];
        let id = mint_client_id("Claude", &uris, SECRET).expect("mint");
        let claims = validate_client_id(&id, SECRET).expect("validate");
        assert_eq!(claims.client_name, "Claude");
        assert_eq!(claims.redirect_uris, uris);
        // Tampered / wrong-secret ids are rejected.
        assert!(validate_client_id(&id, "another-secret-that-is-32-chars!!").is_none());
        assert!(validate_client_id("garbage", SECRET).is_none());
    }

    #[test]
    fn test_client_id_rejects_user_jwts() {
        // A regular user JWT signed with the same secret must NOT be usable
        // as a client_id (different subject).
        let user_jwt =
            crate::auth::jwt::encode_jwt(Uuid::new_v4(), "user@example.com", "User", SECRET, 3600)
                .unwrap();
        assert!(validate_client_id(&user_jwt, SECRET).is_none());
    }

    #[test]
    fn test_redirect_uri_rules() {
        assert!(redirect_uri_allowed(
            "https://claude.ai/api/mcp/auth_callback"
        ));
        assert!(redirect_uri_allowed("http://localhost:8123/callback"));
        assert!(redirect_uri_allowed("http://127.0.0.1/cb"));
        assert!(!redirect_uri_allowed("http://evil.example/cb"));
        assert!(!redirect_uri_allowed("ftp://claude.ai/cb"));
        assert!(!redirect_uri_allowed("http://localhost.evil.example/cb"));
    }

    #[test]
    fn test_auth_code_single_use_and_expiry() {
        let entry = AuthCodeEntry {
            client_id: "c".into(),
            redirect_uri: "https://claude.ai/cb".into(),
            code_challenge: "ch".into(),
            user_id: Uuid::new_v4(),
            email: "a@b.c".into(),
            name: "A".into(),
            created: Instant::now(),
        };
        let code = store_auth_code(entry);
        assert!(take_auth_code(&code).is_some(), "first take succeeds");
        assert!(take_auth_code(&code).is_none(), "second take fails");
        assert!(take_auth_code("nonexistent").is_none());
    }

    #[test]
    fn test_urlencode() {
        assert_eq!(urlencode("abc-._~XYZ09"), "abc-._~XYZ09");
        assert_eq!(urlencode("a b&c=d"), "a%20b%26c%3Dd");
        assert_eq!(
            urlencode("https://x.y/z?a=1"),
            "https%3A%2F%2Fx.y%2Fz%3Fa%3D1"
        );
    }
}
