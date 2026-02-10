//! Auth middleware for Axum routes.
//!
//! Validates JWT Bearer tokens and injects Claims into request extensions.
//! Implements deny-by-default: if `auth_config` is None, all requests are rejected.

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::decode_jwt;
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

/// Middleware that requires a valid JWT Bearer token.
///
/// # Behavior
/// 1. If `auth_config` is `None` → 403 Forbidden (deny-by-default)
/// 2. Extract `Authorization: Bearer <token>` header → 401 if missing
/// 3. Validate JWT with the configured secret → 401 if invalid/expired
/// 4. Check `allowed_email_domain` if configured → 403 if domain mismatch
/// 5. Inject `Claims` into request extensions for downstream handlers
pub async fn require_auth(
    State(state): State<OrchestratorState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AppError> {
    // 1. Deny-by-default if no auth config
    let auth_config = state.auth_config.as_ref().ok_or_else(|| {
        AppError::Forbidden("Authentication not configured — access denied".to_string())
    })?;

    // 2. Extract Bearer token from Authorization header
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("Missing Authorization header".to_string()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AppError::Unauthorized("Invalid Authorization header format".to_string()))?;

    // 3. Decode and validate JWT
    let claims = decode_jwt(token, &auth_config.jwt_secret)
        .map_err(|e| AppError::Unauthorized(format!("Invalid token: {}", e)))?;

    // 4. Check allowed email domain (if configured)
    if let Some(ref domain) = auth_config.allowed_email_domain {
        if !claims.email.ends_with(&format!("@{}", domain)) {
            return Err(AppError::Forbidden(format!(
                "Email domain not allowed (expected @{})",
                domain
            )));
        }
    }

    // 5. Inject claims into request extensions
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::jwt::{encode_jwt, Claims};
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::mock_app_state;
    use crate::AuthConfig;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // for `oneshot`

    const TEST_SECRET: &str = "test-secret-key-minimum-32-chars!!";

    fn test_auth_config() -> AuthConfig {
        AuthConfig {
            google_client_id: "test".to_string(),
            google_client_secret: "test".to_string(),
            google_redirect_uri: "http://localhost/callback".to_string(),
            jwt_secret: TEST_SECRET.to_string(),
            jwt_expiry_secs: 3600,
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

    /// Build a test router with the auth middleware applied
    async fn test_app(auth_config: Option<AuthConfig>) -> Router {
        let state = make_server_state(auth_config).await;

        // Simple handler that returns 200 OK
        async fn ok_handler() -> &'static str {
            "ok"
        }

        Router::new()
            .route("/test", get(ok_handler))
            .layer(from_fn_with_state(state.clone(), require_auth))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_no_auth_config_returns_403() {
        let app = test_app(None).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_no_header_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_invalid_token_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", "Bearer invalid.token.here")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_expired_token_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        // Craft an expired token
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "test@ffs.holdings".to_string(),
            name: "Test".to_string(),
            iat: now - 7200,
            exp: now - 3600,
        };
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(TEST_SECRET.as_bytes()),
        )
        .unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_valid_token_passes() {
        let app = test_app(Some(test_auth_config())).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_wrong_domain_returns_403() {
        let mut config = test_auth_config();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_app(Some(config)).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@gmail.com", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_correct_domain_passes() {
        let mut config = test_auth_config();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_app(Some(config)).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
