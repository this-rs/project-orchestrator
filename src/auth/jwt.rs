//! JWT token encoding and decoding using HS256.
//!
//! The JWT contains user identity claims and is used for both
//! HTTP API authentication (Bearer header) and WebSocket auth
//! (first message after connection).

use anyhow::{Context, Result};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Deterministic UUID for the anonymous user (no-auth mode).
/// Generated from Uuid::nil() — always `00000000-0000-0000-0000-000000000000`.
pub const ANONYMOUS_USER_ID: Uuid = Uuid::nil();

/// JWT claims payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — user UUID
    pub sub: String,
    /// User email
    pub email: String,
    /// User display name
    pub name: String,
    /// Issued at (Unix timestamp)
    pub iat: i64,
    /// Expiration (Unix timestamp)
    pub exp: i64,
}

impl Claims {
    /// Create anonymous claims for no-auth mode.
    ///
    /// Uses a deterministic nil UUID so the anonymous user is always
    /// the same across requests.
    pub fn anonymous() -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            sub: ANONYMOUS_USER_ID.to_string(),
            email: "anonymous@local".to_string(),
            name: "Anonymous".to_string(),
            iat: now,
            exp: now + 86400 * 365 * 100, // effectively never expires
        }
    }
}

/// Encode a JWT token for the given user.
///
/// Uses HS256 signing with the provided secret.
pub fn encode_jwt(
    user_id: Uuid,
    email: &str,
    name: &str,
    secret: &str,
    expiry_secs: u64,
) -> Result<String> {
    let now = chrono::Utc::now().timestamp();
    let claims = Claims {
        sub: user_id.to_string(),
        email: email.to_string(),
        name: name.to_string(),
        iat: now,
        exp: now + expiry_secs as i64,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to encode JWT")
}

/// Decode and validate a JWT token.
///
/// Returns the claims if the token is valid, not expired, and
/// signed with the correct secret.
pub fn decode_jwt(token: &str, secret: &str) -> Result<Claims> {
    let token_data: TokenData<Claims> = decode(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .context("Failed to decode JWT")?;

    Ok(token_data.claims)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SECRET: &str = "test-secret-key-minimum-32-chars!!";

    #[test]
    fn test_encode_decode_roundtrip() {
        let user_id = Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600)
            .expect("encode should succeed");

        let claims = decode_jwt(&token, TEST_SECRET).expect("decode should succeed");
        assert_eq!(claims.sub, user_id.to_string());
        assert_eq!(claims.email, "alice@ffs.holdings");
        assert_eq!(claims.name, "Alice");
        assert!(claims.exp > claims.iat);
        assert_eq!(claims.exp - claims.iat, 3600);
    }

    #[test]
    fn test_expired_token_rejected() {
        // Manually craft a token with exp in the past
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            sub: Uuid::new_v4().to_string(),
            email: "bob@ffs.holdings".to_string(),
            name: "Bob".to_string(),
            iat: now - 7200, // issued 2h ago
            exp: now - 3600, // expired 1h ago
        };

        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(TEST_SECRET.as_bytes()),
        )
        .expect("encode should succeed");

        let result = decode_jwt(&token, TEST_SECRET);
        assert!(result.is_err(), "expired token should be rejected");
    }

    #[test]
    fn test_invalid_secret_rejected() {
        let user_id = Uuid::new_v4();
        let token = encode_jwt(
            user_id,
            "charlie@ffs.holdings",
            "Charlie",
            TEST_SECRET,
            3600,
        )
        .expect("encode should succeed");

        let result = decode_jwt(&token, "wrong-secret-that-is-also-32chars!");
        assert!(result.is_err(), "wrong secret should be rejected");
    }

    #[test]
    fn test_malformed_token_rejected() {
        let result = decode_jwt("not.a.valid.jwt", TEST_SECRET);
        assert!(result.is_err(), "malformed token should be rejected");

        let result = decode_jwt("", TEST_SECRET);
        assert!(result.is_err(), "empty token should be rejected");

        let result = decode_jwt("just-random-text", TEST_SECRET);
        assert!(result.is_err(), "random text should be rejected");
    }

    #[test]
    fn test_claims_sub_is_valid_uuid() {
        let user_id = Uuid::new_v4();
        let token = encode_jwt(user_id, "test@ffs.holdings", "Test", TEST_SECRET, 3600)
            .expect("encode should succeed");

        let claims = decode_jwt(&token, TEST_SECRET).expect("decode should succeed");
        let parsed: Uuid = claims.sub.parse().expect("sub should be a valid UUID");
        assert_eq!(parsed, user_id);
    }
}
