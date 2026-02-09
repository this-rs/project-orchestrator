//! AuthUser extractor for Axum handlers.
//!
//! Extracts the authenticated user's identity from request extensions
//! (populated by the `require_auth` middleware).

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::Claims;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use uuid::Uuid;

/// Authenticated user identity extracted from JWT claims.
///
/// Use this as a handler parameter to require authentication
/// and access the user's identity:
///
/// ```rust,ignore
/// async fn my_handler(user: AuthUser) -> impl IntoResponse {
///     format!("Hello, {}!", user.name)
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: Uuid,
    pub email: String,
    pub name: String,
}

impl AuthUser {
    /// Create from JWT claims
    fn from_claims(claims: &Claims) -> Result<Self, AppError> {
        let user_id: Uuid = claims
            .sub
            .parse()
            .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

        Ok(Self {
            user_id,
            email: claims.email.clone(),
            name: claims.name.clone(),
        })
    }
}

impl FromRequestParts<OrchestratorState> for AuthUser {
    type Rejection = AppError;

    fn from_request_parts(
        parts: &mut Parts,
        _state: &OrchestratorState,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        async {
            let claims = parts
                .extensions
                .get::<Claims>()
                .ok_or_else(|| {
                    AppError::Unauthorized(
                        "Authentication required — no claims in request".to_string(),
                    )
                })?;

            Self::from_claims(claims)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_user_from_valid_claims() {
        let user_id = Uuid::new_v4();
        let claims = Claims {
            sub: user_id.to_string(),
            email: "alice@ffs.holdings".to_string(),
            name: "Alice".to_string(),
            iat: 0,
            exp: 0,
        };

        let user = AuthUser::from_claims(&claims).unwrap();
        assert_eq!(user.user_id, user_id);
        assert_eq!(user.email, "alice@ffs.holdings");
        assert_eq!(user.name, "Alice");
    }

    #[test]
    fn test_auth_user_from_invalid_uuid() {
        let claims = Claims {
            sub: "not-a-uuid".to_string(),
            email: "alice@ffs.holdings".to_string(),
            name: "Alice".to_string(),
            iat: 0,
            exp: 0,
        };

        let result = AuthUser::from_claims(&claims);
        assert!(result.is_err());
    }
}
