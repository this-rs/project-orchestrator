//! WebSocket authentication via first-message handshake.
//!
//! Protocol:
//! 1. Client connects to WebSocket (no HTTP auth required)
//! 2. Server waits for the first message: `{ "type": "auth", "token": "..." }`
//! 3. Server validates the JWT token
//! 4. If valid → sends `{ "type": "auth_ok", "user": {...} }` and returns Claims
//! 5. If invalid → sends `{ "type": "auth_error", "message": "..." }` and closes
//!
//! The token is validated ONCE at connection time. Even if it expires during
//! an active stream, the connection remains open.

use crate::auth::jwt::{decode_jwt, Claims};
use crate::AuthConfig;
use axum::extract::ws::{Message, WebSocket};
use futures::StreamExt;
use serde::Deserialize;
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};

/// Default timeout (in seconds) for the client to send the auth message
const AUTH_TIMEOUT_SECS: u64 = 10;

/// Expected shape of the auth message from the client
#[derive(Debug, Deserialize)]
struct WsAuthMessage {
    #[serde(rename = "type")]
    msg_type: String,
    token: Option<String>,
}

/// Authenticate a WebSocket connection via the first message.
///
/// Returns the validated `Claims` on success, or an error message string
/// on failure. On failure, an `auth_error` message is sent to the client
/// before returning.
///
/// # Behavior
/// - If `auth_config` is `None` → immediate reject (deny-by-default)
/// - Waits up to `AUTH_TIMEOUT_SECS` for the first message
/// - Expects `{ "type": "auth", "token": "<jwt>" }`
/// - Validates the JWT and optionally checks the email domain
/// - On success, sends `{ "type": "auth_ok", "user": {...} }`
pub async fn ws_authenticate(
    socket: &mut WebSocket,
    auth_config: &Option<AuthConfig>,
) -> Result<Claims, String> {
    // 1. Deny-by-default if no auth config
    let config = match auth_config {
        Some(c) => c,
        None => {
            send_auth_error(socket, "Authentication not configured — access denied").await;
            return Err("No auth config".to_string());
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

    // 6. Check email domain restriction (if configured)
    if let Some(ref domain) = config.allowed_email_domain {
        if !claims.email.ends_with(&format!("@{}", domain)) {
            send_auth_error(
                socket,
                &format!("Email domain not allowed (expected @{})", domain),
            )
            .await;
            return Err("Email domain not allowed".to_string());
        }
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
    use crate::test_helpers::test_auth_config;
    use uuid::Uuid;

    // Helper to make a valid auth message JSON
    fn auth_message(token: &str) -> String {
        serde_json::json!({
            "type": "auth",
            "token": token
        })
        .to_string()
    }

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
