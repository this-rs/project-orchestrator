//! Generic OIDC (OpenID Connect) client.
//!
//! Supports any OIDC-compliant provider: Google, Microsoft, Okta, Keycloak, etc.
//! Can be configured with explicit endpoints or via discovery URL.
//!
//! For backward compatibility, legacy `google_*` config fields are mapped to
//! an OidcClient with hardcoded Google endpoints.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::{AuthConfig, OidcConfig};

// Google OIDC endpoints (used for legacy config compatibility)
const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v3/userinfo";

/// User information retrieved from an OIDC provider after successful auth.
#[derive(Debug, Clone, Deserialize)]
pub struct OidcUserInfo {
    /// Provider's unique user identifier (the "sub" claim)
    #[serde(rename = "sub")]
    pub external_id: String,
    /// User's email address
    pub email: String,
    /// User's display name
    pub name: String,
    /// URL to the user's profile picture
    pub picture: Option<String>,
}

/// OIDC discovery document (subset of fields we need).
#[derive(Debug, Deserialize)]
struct DiscoveryDocument {
    authorization_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: Option<String>,
}

/// Token endpoint response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[allow(dead_code)]
    token_type: String,
    #[allow(dead_code)]
    expires_in: Option<u64>,
}

/// Generic OIDC client for authorization code flow.
pub struct OidcClient {
    pub provider_name: String,
    auth_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: Option<String>,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    scopes: String,
    http_client: reqwest::Client,
}

impl OidcClient {
    /// Create from an explicit `OidcConfig`.
    ///
    /// Uses auth_endpoint/token_endpoint from config (must be present).
    pub fn from_config(config: &OidcConfig) -> Result<Self> {
        let auth_endpoint = config
            .auth_endpoint
            .clone()
            .ok_or_else(|| anyhow::anyhow!("OIDC auth_endpoint is required when discovery_url is not set"))?;
        let token_endpoint = config
            .token_endpoint
            .clone()
            .ok_or_else(|| anyhow::anyhow!("OIDC token_endpoint is required when discovery_url is not set"))?;

        Ok(Self {
            provider_name: config.provider_name.clone(),
            auth_endpoint,
            token_endpoint,
            userinfo_endpoint: config.userinfo_endpoint.clone(),
            client_id: config.client_id.clone(),
            client_secret: config.client_secret.clone(),
            redirect_uri: config.redirect_uri.clone(),
            scopes: config.scopes.clone(),
            http_client: reqwest::Client::new(),
        })
    }

    /// Create from a discovery URL (fetches .well-known/openid-configuration).
    pub async fn from_discovery(config: &OidcConfig) -> Result<Self> {
        let discovery_url = config
            .discovery_url
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("discovery_url is required for from_discovery()"))?;

        let client = reqwest::Client::new();
        let doc: DiscoveryDocument = client
            .get(discovery_url)
            .send()
            .await
            .context("Failed to fetch OIDC discovery document")?
            .json()
            .await
            .context("Failed to parse OIDC discovery document")?;

        Ok(Self {
            provider_name: config.provider_name.clone(),
            auth_endpoint: doc.authorization_endpoint,
            token_endpoint: doc.token_endpoint,
            userinfo_endpoint: doc.userinfo_endpoint.or_else(|| config.userinfo_endpoint.clone()),
            client_id: config.client_id.clone(),
            client_secret: config.client_secret.clone(),
            redirect_uri: config.redirect_uri.clone(),
            scopes: config.scopes.clone(),
            http_client: client,
        })
    }

    /// Create from legacy `google_*` fields in AuthConfig.
    ///
    /// Maps the old Google-specific config to a generic OIDC client
    /// with hardcoded Google endpoints.
    pub fn from_legacy_google(auth_config: &AuthConfig) -> Result<Self> {
        let client_id = auth_config
            .google_client_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("google_client_id is required"))?;
        let client_secret = auth_config
            .google_client_secret
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("google_client_secret is required"))?;
        let redirect_uri = auth_config
            .google_redirect_uri
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("google_redirect_uri is required"))?;

        Ok(Self {
            provider_name: "Google".to_string(),
            auth_endpoint: GOOGLE_AUTH_URL.to_string(),
            token_endpoint: GOOGLE_TOKEN_URL.to_string(),
            userinfo_endpoint: Some(GOOGLE_USERINFO_URL.to_string()),
            client_id: client_id.clone(),
            client_secret: client_secret.clone(),
            redirect_uri: redirect_uri.clone(),
            scopes: "openid email profile".to_string(),
            http_client: reqwest::Client::new(),
        })
    }

    /// Create from AuthConfig using the best available method.
    ///
    /// Priority:
    /// 1. Explicit `oidc` section with discovery_url → fetch discovery
    /// 2. Explicit `oidc` section with endpoints → use directly
    /// 3. Legacy `google_*` fields → map to Google OIDC
    pub async fn from_auth_config(auth_config: &AuthConfig) -> Result<Self> {
        if let Some(ref oidc) = auth_config.oidc {
            if oidc.discovery_url.is_some() {
                return Self::from_discovery(oidc).await;
            }
            return Self::from_config(oidc);
        }

        // Fall back to legacy Google fields
        Self::from_legacy_google(auth_config)
    }

    /// Create from AuthConfig synchronously (no discovery fetch).
    ///
    /// Uses explicit endpoints or legacy Google fields. Fails if only
    /// discovery_url is provided without explicit endpoints.
    pub fn from_auth_config_sync(auth_config: &AuthConfig) -> Result<Self> {
        if let Some(ref oidc) = auth_config.oidc {
            if oidc.auth_endpoint.is_some() && oidc.token_endpoint.is_some() {
                return Self::from_config(oidc);
            }
            if oidc.discovery_url.is_some() {
                bail!("Cannot create OidcClient synchronously with discovery_url — use from_auth_config() instead");
            }
            bail!("OIDC config requires either discovery_url or both auth_endpoint and token_endpoint");
        }

        Self::from_legacy_google(auth_config)
    }

    /// Generate the OIDC authorization URL.
    ///
    /// The user should be redirected to this URL to initiate the OAuth flow.
    pub fn auth_url(&self) -> String {
        format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
            self.auth_endpoint,
            urlencoding::encode(&self.client_id),
            urlencoding::encode(&self.redirect_uri),
            urlencoding::encode(&self.scopes),
        )
    }

    /// Exchange an authorization code for user information.
    ///
    /// 1. POST to token endpoint to exchange code for access token
    /// 2. GET userinfo endpoint with access token to retrieve user details
    pub async fn exchange_code(&self, code: &str) -> Result<OidcUserInfo> {
        // Step 1: Exchange code for access token
        let token_response = self
            .http_client
            .post(&self.token_endpoint)
            .form(&[
                ("code", code),
                ("client_id", &self.client_id),
                ("client_secret", &self.client_secret),
                ("redirect_uri", &self.redirect_uri),
                ("grant_type", "authorization_code"),
            ])
            .send()
            .await
            .context("Failed to request OIDC token")?;

        if !token_response.status().is_success() {
            let status = token_response.status();
            let body = token_response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            bail!("OIDC token exchange failed ({}): {}", status, body);
        }

        let token: TokenResponse = token_response
            .json()
            .await
            .context("Failed to parse OIDC token response")?;

        // Step 2: Fetch user info
        let userinfo_url = self
            .userinfo_endpoint
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No userinfo endpoint configured"))?;

        let userinfo_response = self
            .http_client
            .get(userinfo_url)
            .bearer_auth(&token.access_token)
            .send()
            .await
            .context("Failed to request OIDC userinfo")?;

        if !userinfo_response.status().is_success() {
            let status = userinfo_response.status();
            let body = userinfo_response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            bail!("OIDC userinfo fetch failed ({}): {}", status, body);
        }

        let user_info: OidcUserInfo = userinfo_response
            .json()
            .await
            .context("Failed to parse OIDC userinfo response")?;

        Ok(user_info)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::test_auth_config;

    #[test]
    fn test_oidc_from_legacy_google() {
        let config = test_auth_config();
        let client = OidcClient::from_legacy_google(&config).unwrap();

        assert_eq!(client.provider_name, "Google");
        assert_eq!(client.auth_endpoint, GOOGLE_AUTH_URL);
        assert_eq!(client.token_endpoint, GOOGLE_TOKEN_URL);
        assert_eq!(
            client.userinfo_endpoint.as_deref(),
            Some(GOOGLE_USERINFO_URL)
        );
    }

    #[test]
    fn test_oidc_auth_url_google_compat() {
        let config = test_auth_config();
        let client = OidcClient::from_legacy_google(&config).unwrap();
        let url = client.auth_url();

        assert!(url.starts_with(GOOGLE_AUTH_URL));
        assert!(url.contains("client_id="));
        assert!(url.contains("redirect_uri="));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("scope=openid"));
    }

    #[test]
    fn test_oidc_from_explicit_config() {
        let config = OidcConfig {
            provider_name: "Okta".to_string(),
            client_id: "okta-id".to_string(),
            client_secret: "okta-secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some("https://okta.example.com/authorize".to_string()),
            token_endpoint: Some("https://okta.example.com/token".to_string()),
            userinfo_endpoint: Some("https://okta.example.com/userinfo".to_string()),
            scopes: "openid email profile".to_string(),
            discovery_url: None,
        };

        let client = OidcClient::from_config(&config).unwrap();
        assert_eq!(client.provider_name, "Okta");
        assert_eq!(client.auth_endpoint, "https://okta.example.com/authorize");

        let url = client.auth_url();
        assert!(url.starts_with("https://okta.example.com/authorize"));
        assert!(url.contains("client_id=okta-id"));
    }

    #[test]
    fn test_oidc_from_auth_config_sync_legacy() {
        let config = test_auth_config();
        let client = OidcClient::from_auth_config_sync(&config).unwrap();
        assert_eq!(client.provider_name, "Google");
    }

    #[test]
    fn test_oidc_from_auth_config_sync_explicit() {
        let config = AuthConfig {
            jwt_secret: "test-secret-key-minimum-32-chars!!".to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: None,
            frontend_url: None,
            allow_registration: false,
            root_account: None,
            oidc: Some(OidcConfig {
                provider_name: "Microsoft".to_string(),
                client_id: "ms-id".to_string(),
                client_secret: "ms-secret".to_string(),
                redirect_uri: "http://localhost/callback".to_string(),
                auth_endpoint: Some("https://login.microsoftonline.com/authorize".to_string()),
                token_endpoint: Some("https://login.microsoftonline.com/token".to_string()),
                userinfo_endpoint: None,
                scopes: "openid email profile".to_string(),
                discovery_url: None,
            }),
            google_client_id: None,
            google_client_secret: None,
            google_redirect_uri: None,
        };

        let client = OidcClient::from_auth_config_sync(&config).unwrap();
        assert_eq!(client.provider_name, "Microsoft");
        assert!(client.auth_url().contains("login.microsoftonline.com"));
    }

    #[test]
    fn test_oidc_userinfo_deserialization() {
        let json = r#"{
            "sub": "1234567890",
            "email": "alice@company.com",
            "name": "Alice Dupont",
            "picture": "https://example.com/photo.jpg"
        }"#;

        let user: OidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(user.external_id, "1234567890");
        assert_eq!(user.email, "alice@company.com");
        assert_eq!(user.name, "Alice Dupont");
        assert_eq!(user.picture.as_deref(), Some("https://example.com/photo.jpg"));
    }

    #[test]
    fn test_oidc_userinfo_without_picture() {
        let json = r#"{
            "sub": "1234567890",
            "email": "bob@company.com",
            "name": "Bob"
        }"#;

        let user: OidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(user.external_id, "1234567890");
        assert!(user.picture.is_none());
    }

    #[test]
    fn test_oidc_discovery_document_parsing() {
        let json = r#"{
            "issuer": "https://accounts.google.com",
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs"
        }"#;

        let doc: DiscoveryDocument = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(
            doc.authorization_endpoint,
            "https://accounts.google.com/o/oauth2/v2/auth"
        );
        assert_eq!(
            doc.token_endpoint,
            "https://oauth2.googleapis.com/token"
        );
        assert_eq!(
            doc.userinfo_endpoint.as_deref(),
            Some("https://openidconnect.googleapis.com/v1/userinfo")
        );
    }
}
