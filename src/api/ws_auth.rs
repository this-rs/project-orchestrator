//! WebSocket authentication — dual strategy.
//!
//! ## Strategy 1: Cookie-based auth (before upgrade)
//!
//! For browser clients that send the `refresh_token` cookie automatically:
//! 1. Extract the `refresh_token` cookie from the HTTP upgrade headers
//! 2. Validate the token hash in the database (non-expired, non-revoked)
//! 3. Look up the user to build Claims
//! 4. Accept the WebSocket upgrade → send `auth_ok` immediately
//!
//! This is the preferred path — authentication happens BEFORE the upgrade,
//! so invalid credentials result in a 401 HTTP response (no WS connection).
//!
//! ## Strategy 2: First-message handshake (fallback, retrocompat)
//!
//! For non-browser clients (MCP, CLI) that can't send cookies:
//! 1. Accept the WebSocket upgrade with no auth check
//! 2. Wait for `{ "type": "auth", "token": "<jwt>" }` as the first message
//! 3. Validate the JWT
//! 4. Send `auth_ok` on success, `auth_error` + close on failure
//!
//! The token is validated ONCE at connection time. Even if it expires during
//! an active stream, the connection remains open.

use crate::auth::jwt::{decode_jwt, Claims};
use crate::auth::refresh;
use crate::neo4j::GraphStore;
use crate::AuthConfig;
use axum::extract::ws::{Message, WebSocket};
use futures::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};
use uuid::Uuid;

/// Default timeout (in seconds) for the client to send the auth message
const AUTH_TIMEOUT_SECS: u64 = 10;

/// Expected shape of the auth message from the client
#[derive(Debug, Deserialize)]
struct WsAuthMessage {
    #[serde(rename = "type")]
    msg_type: String,
    token: Option<String>,
}

/// Result of pre-upgrade cookie authentication.
///
/// Used by WebSocket upgrade handlers to decide the upgrade strategy:
/// - `Authenticated(Claims)` → cookie valid, upgrade and send `auth_ok` immediately
/// - `NoCookie` → no cookie present, upgrade and fall back to first-message handshake
/// - `Invalid(String)` → cookie present but invalid/expired/revoked, reject with 401
#[derive(Debug)]
pub enum CookieAuthResult {
    /// Cookie was present and valid — upgrade the connection
    Authenticated(Claims),
    /// No cookie present — upgrade and use first-message handshake (retrocompat)
    NoCookie,
    /// Cookie present but invalid — reject the upgrade with 401
    Invalid(String),
}

/// Authenticate a WebSocket upgrade request via the `refresh_token` cookie.
///
/// This is called BEFORE the WebSocket upgrade to validate cookie-based auth.
/// The result tells the caller whether to:
/// - Accept the upgrade with pre-authenticated Claims
/// - Accept the upgrade and fall back to first-message handshake
/// - Reject the upgrade with 401
///
/// # Behavior
/// 1. If `auth_config` is `None` → **no-auth mode**: return `Authenticated` with anonymous Claims
/// 2. If no `Cookie` header or no `refresh_token` cookie → return `NoCookie` (fallback)
/// 3. If cookie present → validate token hash in DB:
///    - Valid → look up user, return `Authenticated(Claims)`
///    - Invalid/expired/revoked → return `Invalid(reason)`
pub async fn ws_authenticate_from_cookie(
    headers: &axum::http::HeaderMap,
    auth_config: &Option<AuthConfig>,
    neo4j: &Arc<dyn GraphStore>,
) -> CookieAuthResult {
    // 1. No-auth mode → anonymous access
    let config = match auth_config {
        Some(c) => c,
        None => {
            debug!("WS cookie auth: no-auth mode → anonymous");
            return CookieAuthResult::Authenticated(Claims::anonymous());
        }
    };

    // 2. Extract cookie from HTTP headers
    let cookie_header = match headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
    {
        Some(h) => h,
        None => {
            debug!("WS cookie auth: no Cookie header → fallback to message auth");
            return CookieAuthResult::NoCookie;
        }
    };

    let raw_token = match refresh::extract_refresh_token_from_cookie(cookie_header) {
        Some(t) => t,
        None => {
            debug!("WS cookie auth: no refresh_token in cookie → fallback to message auth");
            return CookieAuthResult::NoCookie;
        }
    };

    // 3. Validate the refresh token in DB (checks expiry + revoked)
    let token_hash = refresh::hash_token(&raw_token);
    let token_node = match neo4j.validate_refresh_token(&token_hash).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            warn!("WS cookie auth: invalid or expired refresh token");
            return CookieAuthResult::Invalid(
                "Invalid or expired refresh token".to_string(),
            );
        }
        Err(e) => {
            warn!("WS cookie auth: DB error validating token: {}", e);
            return CookieAuthResult::Invalid(format!("Token validation error: {}", e));
        }
    };

    // 4. Look up user info to build Claims
    let (user_id, email, name) = match neo4j.get_user_by_id(token_node.user_id).await {
        Ok(Some(user)) => (user.id, user.email, user.name),
        Ok(None) => {
            // Root account fallback — root users are not stored in Neo4j
            if let Some(ref root) = config.root_account {
                let root_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, root.email.as_bytes());
                if root_id == token_node.user_id {
                    (root_id, root.email.clone(), root.name.clone())
                } else {
                    warn!("WS cookie auth: user not found for token");
                    return CookieAuthResult::Invalid("User not found".to_string());
                }
            } else {
                warn!("WS cookie auth: user not found for token");
                return CookieAuthResult::Invalid("User not found".to_string());
            }
        }
        Err(e) => {
            warn!("WS cookie auth: DB error looking up user: {}", e);
            return CookieAuthResult::Invalid(format!("User lookup error: {}", e));
        }
    };

    // 5. Build Claims (use a long expiry since WS connections persist)
    let now = chrono::Utc::now().timestamp();
    let claims = Claims {
        sub: user_id.to_string(),
        email: email.clone(),
        name: name.clone(),
        iat: now,
        exp: now + config.access_token_expiry_secs as i64,
    };

    debug!(email = %email, "WS cookie auth: authenticated via refresh_token cookie");
    CookieAuthResult::Authenticated(claims)
}

/// Authenticate a WebSocket connection via the first message.
///
/// Returns the validated `Claims` on success, or an error message string
/// on failure. On failure, an `auth_error` message is sent to the client
/// before returning.
///
/// # Behavior
/// 1. If `auth_config` is `None` → **open access** (no-auth mode):
///    send `auth_ok` with anonymous user and return anonymous Claims immediately.
/// 2. If `auth_config` is `Some(...)` → **JWT required**:
///    a. Wait up to `AUTH_TIMEOUT_SECS` for the first message
///    b. Expect `{ "type": "auth", "token": "<jwt>" }`
///    c. Validate the JWT and optionally check the email domain
///    d. On success, send `{ "type": "auth_ok", "user": {...} }`
pub async fn ws_authenticate(
    socket: &mut WebSocket,
    auth_config: &Option<AuthConfig>,
) -> Result<Claims, String> {
    // 1. No-auth mode: send auth_ok with anonymous user and return immediately
    let config = match auth_config {
        Some(c) => c,
        None => {
            let claims = Claims::anonymous();
            let auth_ok = serde_json::json!({
                "type": "auth_ok",
                "user": {
                    "id": claims.sub,
                    "email": claims.email,
                    "name": claims.name,
                }
            });
            let _ = socket.send(Message::Text(auth_ok.to_string().into())).await;
            debug!("WebSocket authenticated (anonymous — no-auth mode)");
            return Ok(claims);
        }
    };

    // 2. Wait for the first message with a timeout
    let first_msg = match timeout(Duration::from_secs(AUTH_TIMEOUT_SECS), socket.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => text.to_string(),
        Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
            debug!("WebSocket closed before auth message");
            return Err("Connection closed before auth".to_string());
        }
        Ok(Some(Ok(_))) => {
            send_auth_error(socket, "Expected text message with auth token").await;
            return Err("Non-text message received".to_string());
        }
        Ok(Some(Err(e))) => {
            warn!("WebSocket error during auth: {}", e);
            return Err(format!("WebSocket error: {}", e));
        }
        Err(_) => {
            send_auth_error(
                socket,
                "Authentication timeout — no auth message received within 10s",
            )
            .await;
            return Err("Auth timeout".to_string());
        }
    };

    // 3. Parse the auth message
    let auth_msg: WsAuthMessage = match serde_json::from_str(&first_msg) {
        Ok(msg) => msg,
        Err(e) => {
            send_auth_error(socket, &format!("Invalid auth message format: {}", e)).await;
            return Err(format!("Invalid format: {}", e));
        }
    };

    // 4. Validate message type
    if auth_msg.msg_type != "auth" {
        send_auth_error(
            socket,
            &format!(
                "Expected message type \"auth\", got \"{}\"",
                auth_msg.msg_type
            ),
        )
        .await;
        return Err(format!("Wrong message type: {}", auth_msg.msg_type));
    }

    // 5. Extract and validate the token
    let token = match auth_msg.token {
        Some(t) if !t.is_empty() => t,
        _ => {
            send_auth_error(socket, "Missing or empty token field").await;
            return Err("Missing token".to_string());
        }
    };

    let claims = match decode_jwt(&token, &config.jwt_secret) {
        Ok(c) => c,
        Err(e) => {
            send_auth_error(socket, &format!("Invalid token: {}", e)).await;
            return Err(format!("Invalid token: {}", e));
        }
    };

    // 6. Check email restrictions (domain + individual whitelist)
    if !config.is_email_allowed(&claims.email) {
        send_auth_error(socket, "Email not allowed by server policy").await;
        return Err("Email not allowed by server policy".to_string());
    }

    // 7. Send auth_ok with user info
    let auth_ok = serde_json::json!({
        "type": "auth_ok",
        "user": {
            "id": claims.sub,
            "email": claims.email,
            "name": claims.name,
        }
    });
    let _ = socket.send(Message::Text(auth_ok.to_string().into())).await;

    debug!(email = %claims.email, "WebSocket authenticated");
    Ok(claims)
}

/// Send an `auth_ok` message with user info through the WebSocket.
///
/// Called after upgrade when authentication was done pre-upgrade (cookie)
/// or post-upgrade (first-message handshake).
pub async fn send_auth_ok(socket: &mut WebSocket, claims: &Claims) {
    let auth_ok = serde_json::json!({
        "type": "auth_ok",
        "user": {
            "id": claims.sub,
            "email": claims.email,
            "name": claims.name,
        }
    });
    let _ = socket.send(Message::Text(auth_ok.to_string().into())).await;
}

/// Send an auth_error message and close the WebSocket.
async fn send_auth_error(socket: &mut WebSocket, message: &str) {
    let error = serde_json::json!({
        "type": "auth_error",
        "message": message,
    });
    let _ = socket.send(Message::Text(error.to_string().into())).await;
    let _ = socket.send(Message::Close(None)).await;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::jwt::encode_jwt;
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::test_auth_config;
    use axum::http::HeaderMap;
    use uuid::Uuid;

    // Helper to make a valid auth message JSON
    fn auth_message(token: &str) -> String {
        serde_json::json!({
            "type": "auth",
            "token": token
        })
        .to_string()
    }

    /// Create a mock Neo4j with a user and a valid refresh token.
    /// Returns (mock, user_id, raw_token).
    async fn setup_mock_with_user_and_token() -> (Arc<MockGraphStore>, Uuid, String) {
        let mock = Arc::new(MockGraphStore::new());

        // Create a user via create_password_user
        let user = mock
            .create_password_user("alice@ffs.holdings", "Alice", "hash")
            .await
            .unwrap();
        let user_id = user.id;

        // Create a refresh token
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = chrono::Utc::now() + chrono::Duration::hours(24);
        mock.create_refresh_token(user_id, &token_hash, expires_at)
            .await
            .unwrap();

        (mock, user_id, raw_token)
    }

    /// Build a HeaderMap with a Cookie header containing the refresh_token.
    fn headers_with_cookie(raw_token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            format!("refresh_token={}", raw_token).parse().unwrap(),
        );
        headers
    }

    // ========================================================================
    // ws_authenticate_from_cookie tests
    // ========================================================================

    #[tokio::test]
    async fn test_cookie_auth_no_auth_mode_returns_anonymous() {
        let mock = Arc::new(MockGraphStore::new());
        let headers = HeaderMap::new();

        let result = ws_authenticate_from_cookie(&headers, &None, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Authenticated(claims) => {
                assert_eq!(claims.email, "anonymous@local");
                assert_eq!(claims.name, "Anonymous");
            }
            other => panic!("Expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_no_cookie_header_returns_no_cookie() {
        let mock = Arc::new(MockGraphStore::new());
        let headers = HeaderMap::new();
        let config = Some(test_auth_config());

        let result = ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::NoCookie => {} // expected
            other => panic!("Expected NoCookie, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_no_refresh_token_in_cookie_returns_no_cookie() {
        let mock = Arc::new(MockGraphStore::new());
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            "session=abc123; other=xyz".parse().unwrap(),
        );
        let config = Some(test_auth_config());

        let result = ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::NoCookie => {} // expected
            other => panic!("Expected NoCookie, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_valid_token_returns_claims() {
        let (mock, user_id, raw_token) = setup_mock_with_user_and_token().await;
        let headers = headers_with_cookie(&raw_token);
        let config = Some(test_auth_config());

        let result =
            ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Authenticated(claims) => {
                assert_eq!(claims.sub, user_id.to_string());
                assert_eq!(claims.email, "alice@ffs.holdings");
                assert_eq!(claims.name, "Alice");
            }
            other => panic!("Expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_invalid_token_returns_invalid() {
        let mock = Arc::new(MockGraphStore::new());
        let headers = headers_with_cookie("nonexistent_token_value");
        let config = Some(test_auth_config());

        let result = ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Invalid(reason) => {
                assert!(reason.contains("Invalid or expired"), "Got: {}", reason);
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_revoked_token_returns_invalid() {
        let (mock, _user_id, raw_token) = setup_mock_with_user_and_token().await;
        // Revoke the token
        let token_hash = refresh::hash_token(&raw_token);
        mock.revoke_refresh_token(&token_hash).await.unwrap();

        let headers = headers_with_cookie(&raw_token);
        let config = Some(test_auth_config());

        let result =
            ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Invalid(reason) => {
                assert!(reason.contains("Invalid or expired"), "Got: {}", reason);
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_root_account_fallback() {
        let mock = Arc::new(MockGraphStore::new());

        // Create a root account config
        let mut config = test_auth_config();
        config.root_account = Some(crate::RootAccountConfig {
            email: "root@ffs.holdings".to_string(),
            name: "Root Admin".to_string(),
            password_hash: "hash".to_string(),
        });
        let root_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, b"root@ffs.holdings");

        // Create a refresh token for root user (no UserNode in DB)
        let raw_token = refresh::generate_token();
        let token_hash = refresh::hash_token(&raw_token);
        let expires_at = chrono::Utc::now() + chrono::Duration::hours(24);
        mock.create_refresh_token(root_id, &token_hash, expires_at)
            .await
            .unwrap();

        let headers = headers_with_cookie(&raw_token);

        let result = ws_authenticate_from_cookie(
            &headers,
            &Some(config),
            &(mock as Arc<dyn GraphStore>),
        )
        .await;

        match result {
            CookieAuthResult::Authenticated(claims) => {
                assert_eq!(claims.sub, root_id.to_string());
                assert_eq!(claims.email, "root@ffs.holdings");
                assert_eq!(claims.name, "Root Admin");
            }
            other => panic!("Expected Authenticated, got {:?}", other),
        }
    }

    // ========================================================================
    // ws_authenticate (first-message handshake) tests
    // ========================================================================

    #[test]
    fn test_ws_auth_message_deserialization() {
        let json = r#"{"type":"auth","token":"eyJ..."}"#;
        let msg: WsAuthMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.msg_type, "auth");
        assert_eq!(msg.token.as_deref(), Some("eyJ..."));
    }

    #[test]
    fn test_ws_auth_message_missing_token() {
        let json = r#"{"type":"auth"}"#;
        let msg: WsAuthMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.msg_type, "auth");
        assert!(msg.token.is_none());
    }

    #[test]
    fn test_ws_auth_message_wrong_type() {
        let json = r#"{"type":"user_message","content":"hello"}"#;
        let msg: WsAuthMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.msg_type, "user_message");
    }

    #[test]
    fn test_valid_auth_message_construction() {
        let user_id = Uuid::new_v4();
        let config = test_auth_config();
        let token = encode_jwt(
            user_id,
            "alice@ffs.holdings",
            "Alice",
            &config.jwt_secret,
            3600,
        )
        .unwrap();
        let msg = auth_message(&token);
        let parsed: WsAuthMessage = serde_json::from_str(&msg).unwrap();
        assert_eq!(parsed.msg_type, "auth");
        assert!(parsed.token.is_some());
    }
}
