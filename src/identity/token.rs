//! PASETO v4.public token creation and verification.
//!
//! Uses PASETO v4.public (Ed25519 signatures) for authenticating P2P messages.
//! Tokens carry claims: issuer (did:key), subject, issued_at, expiration, and
//! custom claims for extensibility.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{SigningKey, VerifyingKey};
use pasetors::claims::{Claims, ClaimsValidationRules};
use pasetors::keys::{AsymmetricKeyPair, AsymmetricPublicKey, AsymmetricSecretKey};
use pasetors::token::UntrustedToken;
use pasetors::version4::V4;
use pasetors::{public, Public};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Claims embedded in a PASETO token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Issuer — the did:key of the signing instance.
    pub iss: String,
    /// Subject — the purpose of this token (e.g., "p2p-sync", "skill-share").
    pub sub: String,
    /// Issued at timestamp.
    pub iat: DateTime<Utc>,
    /// Expiration timestamp.
    pub exp: DateTime<Utc>,
    /// Custom claims for extensibility.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Default token duration (1 hour).
const DEFAULT_TOKEN_DURATION_HOURS: i64 = 1;

/// Create a PASETO v4.public token signed with the given Ed25519 key.
///
/// # Arguments
/// * `signing_key` - Ed25519 private key to sign the token
/// * `did_key` - The did:key identifier of the issuer
/// * `subject` - Purpose of the token (e.g., "p2p-sync")
/// * `duration` - Token validity duration (None = 1 hour default)
/// * `custom_claims` - Additional claims to embed
pub fn create_token(
    signing_key: &SigningKey,
    did_key: &str,
    subject: &str,
    duration: Option<Duration>,
    custom_claims: Option<HashMap<String, serde_json::Value>>,
) -> Result<String> {
    let now = Utc::now();
    let exp = now + duration.unwrap_or_else(|| Duration::hours(DEFAULT_TOKEN_DURATION_HOURS));

    // Build PASETO claims
    let mut claims = Claims::new()?;
    claims.issuer(did_key)?;
    claims.subject(subject)?;
    claims.issued_at(&now.to_rfc3339())?;
    claims.expiration(&exp.to_rfc3339())?;

    // Add custom claims
    if let Some(customs) = &custom_claims {
        for (key, value) in customs {
            claims.add_additional(key, value.clone())?;
        }
    }

    // Convert Ed25519 signing key to pasetors format
    let sk_bytes = signing_key.to_bytes();
    let vk_bytes = signing_key.verifying_key().to_bytes();

    // pasetors expects 64-byte secret key (signing + verifying concatenated)
    let mut full_key = Vec::with_capacity(64);
    full_key.extend_from_slice(&sk_bytes);
    full_key.extend_from_slice(&vk_bytes);

    let secret_key = AsymmetricSecretKey::<V4>::from(&full_key)
        .map_err(|e| anyhow!("Failed to create PASETO secret key: {e}"))?;

    let token = public::sign(&secret_key, &claims, None, None)
        .map_err(|e| anyhow!("Failed to sign PASETO token: {e}"))?;

    Ok(token)
}

/// Verify a PASETO v4.public token and extract its claims.
///
/// # Arguments
/// * `token` - The PASETO token string (starts with "v4.public.")
/// * `verifying_key` - Ed25519 public key of the expected issuer
///
/// # Returns
/// The verified `TokenClaims` if the token is valid and not expired.
pub fn verify_token(token: &str, verifying_key: &VerifyingKey) -> Result<TokenClaims> {
    let pk_bytes = verifying_key.to_bytes();
    let public_key = AsymmetricPublicKey::<V4>::from(&pk_bytes)
        .map_err(|e| anyhow!("Failed to create PASETO public key: {e}"))?;

    // Set up validation rules (validates exp/nbf by default)
    let rules = ClaimsValidationRules::new();

    let untrusted = UntrustedToken::<Public, V4>::try_from(token)
        .map_err(|e| anyhow!("Invalid PASETO token format: {e}"))?;

    let verified = public::verify(&public_key, &untrusted, &rules, None, None)
        .map_err(|e| anyhow!("PASETO token verification failed: {e}"))?;

    // Parse the claims payload
    let claims_json = verified.payload_claims()
        .ok_or_else(|| anyhow!("PASETO token has no claims"))?;

    let iss = claims_json
        .get_claim("iss")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let sub = claims_json
        .get_claim("sub")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let iat = claims_json
        .get_claim("iat")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let exp = claims_json
        .get_claim("exp")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    // Check expiration
    if exp < Utc::now() {
        return Err(anyhow!("Token has expired (exp: {exp})"));
    }

    // Extract custom claims (everything except standard fields)
    let _standard_fields = ["iss", "sub", "iat", "exp", "nbf", "jti", "aud"];
    let mut custom = HashMap::new();

    // Note: pasetors Claims doesn't expose iteration, so custom claims
    // are only available if we re-parse the payload
    // For now, we return the standard claims

    Ok(TokenClaims {
        iss,
        sub,
        iat,
        exp,
        custom,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn test_create_and_verify_token() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let did_key = "did:key:z6MkTest123";

        let token = create_token(&signing_key, did_key, "p2p-sync", None, None)
            .expect("should create token");

        assert!(token.starts_with("v4.public."), "Token should be PASETO v4.public");

        let claims = verify_token(&token, &signing_key.verifying_key())
            .expect("should verify valid token");

        assert_eq!(claims.iss, did_key);
        assert_eq!(claims.sub, "p2p-sync");
        assert!(claims.exp > Utc::now());
    }

    #[test]
    fn test_token_with_custom_duration() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);

        let token = create_token(
            &signing_key,
            "did:key:z6MkTest",
            "test",
            Some(Duration::minutes(5)),
            None,
        )
        .expect("should create token");

        let claims = verify_token(&token, &signing_key.verifying_key()).unwrap();
        let duration = claims.exp - claims.iat;
        assert!(duration.num_minutes() >= 4 && duration.num_minutes() <= 6);
    }

    #[test]
    fn test_token_with_custom_claims() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let mut custom = HashMap::new();
        custom.insert("peer_id".to_string(), serde_json::json!("alice"));
        custom.insert("sync_version".to_string(), serde_json::json!(3));

        let token = create_token(
            &signing_key,
            "did:key:z6MkTest",
            "sync",
            None,
            Some(custom),
        )
        .expect("should create token with custom claims");

        let claims = verify_token(&token, &signing_key.verifying_key()).unwrap();
        assert_eq!(claims.sub, "sync");
    }

    #[test]
    fn test_wrong_key_verification() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let wrong_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);

        let token = create_token(&signing_key, "did:key:z6MkTest", "test", None, None)
            .expect("should create token");

        let result = verify_token(&token, &wrong_key.verifying_key());
        assert!(result.is_err(), "Should reject token signed with different key");
    }

    #[test]
    fn test_invalid_token_format() {
        let key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let result = verify_token("not-a-valid-token", &key.verifying_key());
        assert!(result.is_err());
    }
}
