//! WebSocket authentication — cookie-based (pre-upgrade).
//!
//! For browser clients that send the `refresh_token` cookie automatically:
//! 1. Extract the `refresh_token` cookie from the HTTP upgrade headers
//! 2. Validate the token hash in the database (non-expired, non-revoked)
//! 3. Look up the user to build Claims
//! 4. Accept the WebSocket upgrade → send `auth_ok` immediately
//!
//! Authentication happens BEFORE the upgrade, so invalid credentials result
//! in a 401 HTTP response (no WS connection opened).
//!
//! In no-auth mode, anonymous Claims are returned immediately.

use crate::auth::jwt::Claims;
use crate::auth::refresh;
use crate::neo4j::GraphStore;
use crate::AuthConfig;
use axum::extract::ws::{Message, WebSocket};
use std::sync::Arc;
use tracing::{debug, warn};
use uuid::Uuid;

/// Result of pre-upgrade cookie authentication.
///
/// Used by WebSocket upgrade handlers to decide whether to accept the upgrade:
/// - `Authenticated(Claims)` → cookie valid (or no-auth mode), upgrade and send `auth_ok`
/// - `Invalid(String)` → no cookie or invalid cookie, reject with 401
#[derive(Debug)]
pub enum CookieAuthResult {
    /// Cookie was present and valid — upgrade the connection
    Authenticated(Claims),
    /// No cookie or invalid cookie — reject the upgrade with 401
    Invalid(String),
}

/// Authenticate a WebSocket upgrade request via the `refresh_token` cookie.
///
/// This is called BEFORE the WebSocket upgrade to validate cookie-based auth.
/// The result tells the caller whether to accept or reject the upgrade.
///
/// # Behavior
/// 1. If `auth_config` is `None` → **no-auth mode**: return `Authenticated` with anonymous Claims
/// 2. If no `Cookie` header or no `refresh_token` cookie → return `Invalid` (reject with 401)
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
            debug!("WS cookie auth: no Cookie header → rejecting");
            return CookieAuthResult::Invalid("No refresh_token cookie".to_string());
        }
    };

    let raw_token = match refresh::extract_refresh_token_from_cookie(cookie_header) {
        Some(t) => t,
        None => {
            debug!("WS cookie auth: no refresh_token in cookie → rejecting");
            return CookieAuthResult::Invalid("No refresh_token cookie".to_string());
        }
    };

    // 3. Validate the refresh token in DB (checks expiry + revoked)
    let token_hash = refresh::hash_token(&raw_token);
    let token_node = match neo4j.validate_refresh_token(&token_hash).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            warn!("WS cookie auth: invalid or expired refresh token");
            return CookieAuthResult::Invalid("Invalid or expired refresh token".to_string());
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

/// Send an `auth_ok` message with user info through the WebSocket.
///
/// Called after upgrade when authentication was done pre-upgrade (cookie).
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::test_auth_config;
    use axum::http::HeaderMap;

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

        let result =
            ws_authenticate_from_cookie(&headers, &None, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Authenticated(claims) => {
                assert_eq!(claims.email, "anonymous@local");
                assert_eq!(claims.name, "Anonymous");
            }
            other => panic!("Expected Authenticated, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_no_cookie_header_returns_invalid() {
        let mock = Arc::new(MockGraphStore::new());
        let headers = HeaderMap::new();
        let config = Some(test_auth_config());

        let result =
            ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Invalid(reason) => {
                assert!(
                    reason.contains("No refresh_token cookie"),
                    "Got: {}",
                    reason
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cookie_auth_no_refresh_token_in_cookie_returns_invalid() {
        let mock = Arc::new(MockGraphStore::new());
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            "session=abc123; other=xyz".parse().unwrap(),
        );
        let config = Some(test_auth_config());

        let result =
            ws_authenticate_from_cookie(&headers, &config, &(mock as Arc<dyn GraphStore>)).await;

        match result {
            CookieAuthResult::Invalid(reason) => {
                assert!(
                    reason.contains("No refresh_token cookie"),
                    "Got: {}",
                    reason
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
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

        let result =
            ws_authenticate_from_cookie(&headers, &Some(config), &(mock as Arc<dyn GraphStore>))
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
}
