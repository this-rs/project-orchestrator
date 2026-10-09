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

/// Marker value for `Claims::token_type` identifying MCP access tokens.
pub const TOKEN_TYPE_MCP: &str = "mcp";

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
    /// Token type discriminator. `None` (absent) = regular access JWT;
    /// `Some("mcp")` = long-lived revocable MCP token (the middleware then
    /// checks `jti` against the McpToken revocation store on every request).
    /// Optional + skipped when absent so existing tokens stay valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
    /// Space-separated OAuth-style scopes (MCP tokens: "mcp:read mcp:write").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// JWT ID — revocation lookup key for MCP tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
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
            token_type: None,
            scope: None,
            jti: None,
        }
    }

    /// Create service-account claims for internally spawned agents.
    ///
    /// Used by the plan runner, task delegation, and protocol executor so that
    /// spawned Claude sessions receive a valid JWT and can call protected MCP
    /// tool routes.
    pub fn service_account(identity: &str) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            sub: identity.to_string(),
            email: "runner@system.local".to_string(),
            name: "Service Account".to_string(),
            iat: now,
            exp: now + 86400, // 24 h
            token_type: None,
            scope: None,
            jti: None,
        }
    }

    /// True when these claims describe a long-lived MCP token (which must be
    /// revocation-checked against the McpToken store).
    pub fn is_mcp_token(&self) -> bool {
        self.token_type.as_deref() == Some(TOKEN_TYPE_MCP)
    }

    /// True for the token a chat session's MCP subprocess presents.
    pub fn is_agent_session(&self) -> bool {
        self.token_type.as_deref() == Some(TOKEN_TYPE_AGENT_SESSION)
    }

    /// Whether a person is behind this token — as opposed to an agent: a chat
    /// session's MCP (`agent_session`), a vault token, an MCP access token, or
    /// the standalone MCP server (which signs as the nil system user).
    ///
    /// Operations that widen an agent's own access (unlocking the vault,
    /// granting secrets) must require this, or an agent could grant itself.
    pub fn is_human(&self) -> bool {
        self.token_type.is_none() && self.sub != ANONYMOUS_USER_ID.to_string()
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
        token_type: None,
        scope: None,
        jti: None,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to encode JWT")
}

/// What an `agent_session` token is bound to. Everything here is covered by
/// the signature: an agent can read its own token but cannot edit it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionBinding {
    /// The chat session the token was minted for.
    pub session_id: String,
    /// Permission mode of that session when the token was minted — the policy
    /// ceiling of anything the session spawns.
    pub ceiling: Option<String>,
    /// MCP tool profile of the session (`None` = the default, full profile).
    pub tool_profile: Option<String>,
    /// The session runs on a provider other than Claude Code, or was opened by
    /// such a session (`lineage:third_party` in the scope). A session opened
    /// with this token as its caller gets the restricted profile whatever its
    /// own provider and mode: a third-party model in `trust` may delegate once,
    /// never in a loop.
    pub third_party: bool,
}

const AGENT_SCOPE_SESSION: &str = "session:";
const AGENT_SCOPE_CEILING: &str = "ceiling:";
const AGENT_SCOPE_TOOLS: &str = "tools:";
const AGENT_SCOPE_THIRD_PARTY: &str = "lineage:third_party";

/// Generate the session token handed to a chat session's MCP subprocess
/// (`PO_AUTH_TOKEN`).
///
/// The token carries the user's identity and is BOUND to the session: the
/// session id, its policy ceiling and its tool profile are signed into `scope`
/// (`session:<id> ceiling:<mode> tools:<profile>`), and it carries a `jti` that
/// [`crate::auth::agent_tokens`] must still know — closing the session revokes
/// it. Returns `(token, jti)`; the caller registers the `jti`.
///
/// `binding` is `None` only for callers that have no session (tests, one-shot
/// tools): such a token authenticates but can neither be traced to a session
/// nor spawn one.
pub fn generate_session_token(
    claims: &Claims,
    binding: Option<&AgentSessionBinding>,
    secret: &str,
    expiry_secs: u64,
) -> Result<(String, String)> {
    let now = chrono::Utc::now().timestamp();
    let jti = Uuid::new_v4().to_string();
    let scope = binding.map(|b| {
        let mut parts = vec![format!("{AGENT_SCOPE_SESSION}{}", b.session_id)];
        if let Some(c) = b.ceiling.as_deref().filter(|c| !c.is_empty()) {
            parts.push(format!("{AGENT_SCOPE_CEILING}{c}"));
        }
        if let Some(t) = b.tool_profile.as_deref().filter(|t| !t.is_empty()) {
            parts.push(format!("{AGENT_SCOPE_TOOLS}{t}"));
        }
        if b.third_party {
            parts.push(AGENT_SCOPE_THIRD_PARTY.to_string());
        }
        parts.join(" ")
    });
    let session_claims = Claims {
        sub: claims.sub.clone(),
        email: claims.email.clone(),
        name: claims.name.clone(),
        iat: now,
        exp: now + expiry_secs as i64,
        token_type: Some(TOKEN_TYPE_AGENT_SESSION.to_string()),
        scope,
        jti: Some(jti.clone()),
    };

    let token = encode(
        &Header::default(),
        &session_claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to encode session token")?;
    Ok((token, jti))
}

/// The session an `agent_session` token is bound to — `None` for any other
/// token type, and for an agent token minted without a session.
pub fn agent_session_binding(claims: &Claims) -> Option<AgentSessionBinding> {
    if !claims.is_agent_session() {
        return None;
    }
    let scope = claims.scope.as_deref()?;
    let mut session_id = None;
    let mut ceiling = None;
    let mut tool_profile = None;
    let mut third_party = false;
    for part in scope.split_whitespace() {
        if part == AGENT_SCOPE_THIRD_PARTY {
            third_party = true;
        } else if let Some(v) = part.strip_prefix(AGENT_SCOPE_SESSION) {
            session_id = Some(v.to_string());
        } else if let Some(v) = part.strip_prefix(AGENT_SCOPE_CEILING) {
            ceiling = Some(v.to_string());
        } else if let Some(v) = part.strip_prefix(AGENT_SCOPE_TOOLS) {
            tool_profile = Some(v.to_string());
        }
    }
    Some(AgentSessionBinding {
        session_id: session_id.filter(|s| !s.is_empty())?,
        ceiling,
        tool_profile,
        third_party,
    })
}

/// Token type of the session token given to a chat session's MCP server.
pub const TOKEN_TYPE_AGENT_SESSION: &str = "agent_session";

/// Token type of a vault token (see [`generate_vault_token`]).
pub const VAULT_TOKEN_TYPE: &str = "vault";
const VAULT_SCOPE_PREFIX: &str = "session:";

/// Mint the token an agent session presents to read the secrets granted to it.
///
/// The session id is SIGNED into the token, in `scope`. An agent's environment
/// also carries `PO_SESSION_ID`, but that variable is set by us and editable by
/// the agent — trusting it would let any session claim another's grants. Only
/// what the signature covers can say which session is asking.
///
/// Reuses the existing claim fields (`token_type`, `scope`) rather than adding
/// one, so every token issued before this change still decodes.
pub fn generate_vault_token(
    claims: &Claims,
    session_id: &str,
    secret: &str,
    expiry_secs: u64,
) -> Result<String> {
    let now = chrono::Utc::now().timestamp();
    let vault_claims = Claims {
        sub: claims.sub.clone(),
        email: claims.email.clone(),
        name: claims.name.clone(),
        iat: now,
        exp: now + expiry_secs as i64,
        token_type: Some(VAULT_TOKEN_TYPE.to_string()),
        scope: Some(format!("{VAULT_SCOPE_PREFIX}{session_id}")),
        jti: None,
    };
    encode(
        &Header::default(),
        &vault_claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to encode vault token")
}

/// The session a vault token was issued for — `None` for any other token.
pub fn vault_token_session(claims: &Claims) -> Option<&str> {
    if claims.token_type.as_deref() != Some(VAULT_TOKEN_TYPE) {
        return None;
    }
    claims
        .scope
        .as_deref()?
        .strip_prefix(VAULT_SCOPE_PREFIX)
        .filter(|s| !s.is_empty())
}

/// Encode a long-lived, revocable, scoped MCP access token.
///
/// Returns `(token, jti)` — the caller must persist the `jti` in the
/// McpToken store (Neo4j) so the middleware can revocation-check it.
/// Signed with the same HS256 secret as regular JWTs; distinguished by
/// `token_type = "mcp"`.
pub fn encode_mcp_token(
    user_id: Uuid,
    email: &str,
    name: &str,
    scope: &str,
    secret: &str,
    expiry_secs: u64,
) -> Result<(String, String)> {
    let now = chrono::Utc::now().timestamp();
    let jti = Uuid::new_v4().to_string();
    let claims = Claims {
        sub: user_id.to_string(),
        email: email.to_string(),
        name: name.to_string(),
        iat: now,
        exp: now + expiry_secs as i64,
        token_type: Some(TOKEN_TYPE_MCP.to_string()),
        scope: Some(scope.to_string()),
        jti: Some(jti.clone()),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to encode MCP token")?;
    Ok((token, jti))
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
            exp: now - 3600, // expired 1h ago,
            token_type: None,
            scope: None,
            jti: None,
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
    fn test_generate_session_token_4h() {
        let original = Claims {
            sub: Uuid::new_v4().to_string(),
            email: "alice@ffs.holdings".to_string(),
            name: "Alice".to_string(),
            iat: chrono::Utc::now().timestamp(),
            exp: chrono::Utc::now().timestamp() + 900, // original 15min token,
            token_type: None,
            scope: None,
            jti: None,
        };

        let (token, _jti) =
            generate_session_token(&original, None, TEST_SECRET, 86400).expect("should succeed");
        let decoded = decode_jwt(&token, TEST_SECRET).expect("should decode");

        assert_eq!(decoded.sub, original.sub);
        assert_eq!(decoded.email, original.email);
        assert_eq!(decoded.name, original.name);
        // Session token should have 24h expiry, not the original 15min
        assert_eq!(decoded.exp - decoded.iat, 86400);
    }

    #[test]
    fn test_generate_session_token_validated_by_decode() {
        let claims = Claims {
            sub: Uuid::new_v4().to_string(),
            email: "bob@ffs.holdings".to_string(),
            name: "Bob".to_string(),
            iat: chrono::Utc::now().timestamp(),
            exp: chrono::Utc::now().timestamp() + 900,
            token_type: None,
            scope: None,
            jti: None,
        };

        let (token, _jti) =
            generate_session_token(&claims, None, TEST_SECRET, 3600).expect("should succeed");
        // Same decode function used by require_auth middleware
        let result = decode_jwt(&token, TEST_SECRET);
        assert!(
            result.is_ok(),
            "session token should be valid for require_auth"
        );
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

    #[test]
    fn session_token_is_bound_to_its_session_and_carries_a_jti() {
        let human = Claims::service_account("runner:t");
        let binding = AgentSessionBinding {
            session_id: "11111111-2222-3333-4444-555555555555".to_string(),
            ceiling: Some("acceptEdits".to_string()),
            tool_profile: Some("restricted".to_string()),
            third_party: false,
        };
        let (token, jti) =
            generate_session_token(&human, Some(&binding), TEST_SECRET, 600).unwrap();
        let decoded = decode_jwt(&token, TEST_SECRET).unwrap();
        assert_eq!(decoded.jti.as_deref(), Some(jti.as_str()));
        assert!(decoded.is_agent_session());
        assert!(!decoded.is_human());
        assert_eq!(agent_session_binding(&decoded), Some(binding));
    }

    #[test]
    fn the_third_party_lineage_is_signed_into_the_session_token() {
        let human = Claims::service_account("runner:t");
        let binding = AgentSessionBinding {
            session_id: "11111111-2222-3333-4444-555555555555".to_string(),
            ceiling: Some("bypassPermissions".to_string()),
            tool_profile: Some("full".to_string()),
            third_party: true,
        };
        let (token, _) = generate_session_token(&human, Some(&binding), TEST_SECRET, 600).unwrap();
        let decoded = decode_jwt(&token, TEST_SECRET).unwrap();
        assert_eq!(agent_session_binding(&decoded), Some(binding));
    }

    #[test]
    fn only_an_agent_session_token_yields_a_binding() {
        // A human token whose scope imitates the agent format binds nothing.
        let mut human = Claims::service_account("someone");
        human.scope = Some("session:abc ceiling:bypassPermissions".to_string());
        assert_eq!(agent_session_binding(&human), None);

        // An agent token minted without a session binds nothing either.
        let (token, _) = generate_session_token(&human, None, TEST_SECRET, 600).unwrap();
        let decoded = decode_jwt(&token, TEST_SECRET).unwrap();
        assert_eq!(agent_session_binding(&decoded), None);
    }
}
