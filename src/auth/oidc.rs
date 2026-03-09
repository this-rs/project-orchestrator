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

/// Slugify a provider name into a stable key (e.g. "Google" → "google", "My Okta" → "my-okta").
fn slugify_provider(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Raw OIDC userinfo response — tolerant of provider-specific claim differences.
///
/// Different providers return different claim names:
/// - Google: `sub`, `email`, `name`, `picture`
/// - Cognito: `sub`, `email`, `username`, `cognito:username` (no `name` or `picture`)
/// - Microsoft: `sub`, `email`, `name` (or `preferred_username`)
/// - Okta: `sub`, `email`, `name`, `given_name`, `family_name`
#[derive(Debug, Clone, Deserialize)]
struct RawOidcUserInfo {
    sub: String,
    email: Option<String>,
    name: Option<String>,
    picture: Option<String>,
    /// Cognito uses `username`
    username: Option<String>,
    /// Some providers use `preferred_username`
    preferred_username: Option<String>,
    /// Fallback: build name from given + family
    given_name: Option<String>,
    family_name: Option<String>,
}

/// User information retrieved from an OIDC provider after successful auth.
#[derive(Debug, Clone)]
pub struct OidcUserInfo {
    /// Provider's unique user identifier (the "sub" claim)
    pub external_id: String,
    /// User's email address
    pub email: String,
    /// User's display name
    pub name: String,
    /// URL to the user's profile picture
    pub picture: Option<String>,
}

impl RawOidcUserInfo {
    /// Convert raw provider-specific claims into a normalized OidcUserInfo.
    fn normalize(self) -> Result<OidcUserInfo> {
        let email = self
            .email
            .ok_or_else(|| anyhow::anyhow!("OIDC userinfo response missing 'email' claim"))?;

        // Resolve display name from multiple possible claims
        let name = self
            .name
            .or(self.preferred_username)
            .or(self.username)
            .or_else(|| match (&self.given_name, &self.family_name) {
                (Some(g), Some(f)) => Some(format!("{} {}", g, f)),
                (Some(g), None) => Some(g.clone()),
                (None, Some(f)) => Some(f.clone()),
                _ => None,
            })
            .unwrap_or_else(|| email.split('@').next().unwrap_or(&email).to_string());

        Ok(OidcUserInfo {
            external_id: self.sub,
            email,
            name,
            picture: self.picture,
        })
    }
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
    /// Stable identifier for the provider (e.g. "google", "cognito", "okta").
    /// Used for provider-specific logic instead of fragile URL matching.
    pub provider_key: String,
    pub provider_name: String,
    auth_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: Option<String>,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    scopes: String,
    /// Extra query parameters appended to the authorization URL.
    /// Allows provider-specific params (e.g. `access_type=offline` for Google)
    /// without hardcoding per provider_key.
    extra_auth_params: Vec<(String, String)>,
    http_client: reqwest::Client,
}

impl OidcClient {
    /// Derive `provider_key` from OidcConfig: use explicit key, or slugify provider_name.
    fn derive_provider_key(config: &OidcConfig) -> String {
        config
            .provider_key
            .clone()
            .unwrap_or_else(|| slugify_provider(&config.provider_name))
    }

    /// Build the extra auth params: merge config's explicit map with Google defaults
    /// when the provider_key is "google" and the config doesn't override them.
    fn build_extra_auth_params(
        provider_key: &str,
        config_params: &std::collections::HashMap<String, String>,
    ) -> Vec<(String, String)> {
        let mut params: Vec<(String, String)> = config_params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Google-specific defaults — only applied if not overridden in config
        if provider_key == "google" {
            if !config_params.contains_key("access_type") {
                params.push(("access_type".to_string(), "offline".to_string()));
            }
            if !config_params.contains_key("prompt") {
                params.push(("prompt".to_string(), "consent".to_string()));
            }
        }

        params
    }

    /// Create from an explicit `OidcConfig`.
    ///
    /// Uses auth_endpoint/token_endpoint from config (must be present).
    pub fn from_config(config: &OidcConfig) -> Result<Self> {
        let auth_endpoint = config.auth_endpoint.clone().ok_or_else(|| {
            anyhow::anyhow!("OIDC auth_endpoint is required when discovery_url is not set")
        })?;
        let token_endpoint = config.token_endpoint.clone().ok_or_else(|| {
            anyhow::anyhow!("OIDC token_endpoint is required when discovery_url is not set")
        })?;
        let provider_key = Self::derive_provider_key(config);
        let extra_auth_params =
            Self::build_extra_auth_params(&provider_key, &config.extra_auth_params);

        Ok(Self {
            provider_key,
            provider_name: config.provider_name.clone(),
            auth_endpoint,
            token_endpoint,
            userinfo_endpoint: config.userinfo_endpoint.clone(),
            client_id: config.client_id.clone(),
            client_secret: config.client_secret.clone(),
            redirect_uri: config.redirect_uri.clone(),
            scopes: config.scopes.clone(),
            extra_auth_params,
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

        let provider_key = Self::derive_provider_key(config);
        let extra_auth_params =
            Self::build_extra_auth_params(&provider_key, &config.extra_auth_params);

        Ok(Self {
            provider_key,
            provider_name: config.provider_name.clone(),
            auth_endpoint: doc.authorization_endpoint,
            token_endpoint: doc.token_endpoint,
            userinfo_endpoint: doc
                .userinfo_endpoint
                .or_else(|| config.userinfo_endpoint.clone()),
            client_id: config.client_id.clone(),
            client_secret: config.client_secret.clone(),
            redirect_uri: config.redirect_uri.clone(),
            scopes: config.scopes.clone(),
            extra_auth_params,
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
            provider_key: "google".to_string(),
            provider_name: "Google".to_string(),
            auth_endpoint: GOOGLE_AUTH_URL.to_string(),
            token_endpoint: GOOGLE_TOKEN_URL.to_string(),
            userinfo_endpoint: Some(GOOGLE_USERINFO_URL.to_string()),
            client_id: client_id.clone(),
            client_secret: client_secret.clone(),
            redirect_uri: redirect_uri.clone(),
            scopes: "openid email profile".to_string(),
            extra_auth_params: vec![
                ("access_type".to_string(), "offline".to_string()),
                ("prompt".to_string(), "consent".to_string()),
            ],
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

    /// Generate the OIDC authorization URL using the configured redirect_uri.
    pub fn auth_url(&self) -> String {
        self.auth_url_with_redirect(&self.redirect_uri)
    }

    /// Generate the OIDC authorization URL with a custom redirect_uri.
    ///
    /// Used for dynamic origin-based redirect URIs (e.g. desktop vs web access).
    /// Extra auth params (provider-specific or from config) are appended automatically.
    pub fn auth_url_with_redirect(&self, redirect_uri: &str) -> String {
        let mut url = format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}",
            self.auth_endpoint,
            urlencoding::encode(&self.client_id),
            urlencoding::encode(redirect_uri),
            urlencoding::encode(&self.scopes),
        );

        // Append extra auth params (e.g. access_type=offline for Google)
        for (key, value) in &self.extra_auth_params {
            url.push_str(&format!(
                "&{}={}",
                urlencoding::encode(key),
                urlencoding::encode(value)
            ));
        }

        url
    }

    /// Returns true if this client is configured for Google (legacy or explicit).
    #[cfg(test)]
    fn is_google_provider(&self) -> bool {
        self.provider_key == "google"
    }

    /// Exchange an authorization code for user information using the configured redirect_uri.
    pub async fn exchange_code(&self, code: &str) -> Result<OidcUserInfo> {
        self.exchange_code_with_redirect(code, &self.redirect_uri)
            .await
    }

    /// Exchange an authorization code for user information with a custom redirect_uri.
    ///
    /// The redirect_uri MUST match the one used in the authorization URL (OAuth providers
    /// enforce strict matching between the auth step and the token exchange step).
    ///
    /// 1. POST to token endpoint to exchange code for access token
    /// 2. GET userinfo endpoint with access token to retrieve user details
    pub async fn exchange_code_with_redirect(
        &self,
        code: &str,
        redirect_uri: &str,
    ) -> Result<OidcUserInfo> {
        // Step 1: Exchange code for access token
        let token_response = self
            .http_client
            .post(&self.token_endpoint)
            .form(&[
                ("code", code),
                ("client_id", &self.client_id),
                ("client_secret", &self.client_secret),
                ("redirect_uri", redirect_uri),
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

        // Parse into tolerant RawOidcUserInfo first, then normalize.
        // On parse failure, log safe metadata only (no PII in error messages).
        let content_type = userinfo_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        let body = userinfo_response
            .text()
            .await
            .context("Failed to read OIDC userinfo response body")?;

        let raw: RawOidcUserInfo = match serde_json::from_str(&body) {
            Ok(info) => info,
            Err(e) => {
                // Full body at DEBUG level only (never in prod logs)
                tracing::debug!(
                    provider = %self.provider_name,
                    body_len = body.len(),
                    content_type = %content_type,
                    "OIDC userinfo raw body for debugging: {}",
                    body
                );
                bail!(
                    "Failed to parse OIDC userinfo response (provider: {}, body_len: {}, content_type: {}): {}",
                    self.provider_name, body.len(), content_type, e
                );
            }
        };

        raw.normalize()
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
    fn test_slugify_provider() {
        assert_eq!(slugify_provider("Google"), "google");
        assert_eq!(slugify_provider("My Okta"), "my-okta");
        assert_eq!(slugify_provider("AWS Cognito"), "aws-cognito");
        assert_eq!(slugify_provider("  spaces  "), "spaces");
        assert_eq!(slugify_provider("Auth0"), "auth0");
    }

    #[test]
    fn test_oidc_from_legacy_google() {
        let config = test_auth_config();
        let client = OidcClient::from_legacy_google(&config).unwrap();

        assert_eq!(client.provider_key, "google");
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
            provider_key: None,
            provider_name: "Okta".to_string(),
            client_id: "okta-id".to_string(),
            client_secret: "okta-secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some("https://okta.example.com/authorize".to_string()),
            token_endpoint: Some("https://okta.example.com/token".to_string()),
            userinfo_endpoint: Some("https://okta.example.com/userinfo".to_string()),
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: Default::default(),
        };

        let client = OidcClient::from_config(&config).unwrap();
        assert_eq!(client.provider_key, "okta");
        assert_eq!(client.provider_name, "Okta");
        assert_eq!(client.auth_endpoint, "https://okta.example.com/authorize");

        let url = client.auth_url();
        assert!(url.starts_with("https://okta.example.com/authorize"));
        assert!(url.contains("client_id=okta-id"));
        // Non-Google provider should NOT have access_type or prompt
        assert!(!url.contains("access_type=offline"));
        assert!(!url.contains("prompt=consent"));
    }

    #[test]
    fn test_oidc_from_auth_config_sync_legacy() {
        let config = test_auth_config();
        let client = OidcClient::from_auth_config_sync(&config).unwrap();
        assert_eq!(client.provider_key, "google");
        assert_eq!(client.provider_name, "Google");
    }

    #[test]
    fn test_oidc_from_auth_config_sync_explicit() {
        let config = AuthConfig {
            jwt_secret: "test-secret-key-minimum-32-chars!!".to_string(),
            access_token_expiry_secs: 900,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
            frontend_url: None,
            additional_origins: vec![],
            allow_registration: false,
            root_account: None,
            oidc: Some(OidcConfig {
                provider_key: None,
                provider_name: "Microsoft".to_string(),
                client_id: "ms-id".to_string(),
                client_secret: "ms-secret".to_string(),
                redirect_uri: "http://localhost/callback".to_string(),
                auth_endpoint: Some("https://login.microsoftonline.com/authorize".to_string()),
                token_endpoint: Some("https://login.microsoftonline.com/token".to_string()),
                userinfo_endpoint: None,
                scopes: "openid email profile".to_string(),
                discovery_url: None,
                extra_auth_params: Default::default(),
            }),
            google_client_id: None,
            google_client_secret: None,
            google_redirect_uri: None,
        };

        let client = OidcClient::from_auth_config_sync(&config).unwrap();
        assert_eq!(client.provider_key, "microsoft");
        assert_eq!(client.provider_name, "Microsoft");
        assert!(client.auth_url().contains("login.microsoftonline.com"));
    }

    #[test]
    fn test_oidc_userinfo_google_format() {
        let json = r#"{
            "sub": "1234567890",
            "email": "alice@company.com",
            "name": "Alice Dupont",
            "picture": "https://example.com/photo.jpg"
        }"#;

        let raw: RawOidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        let user = raw.normalize().expect("should normalize");
        assert_eq!(user.external_id, "1234567890");
        assert_eq!(user.email, "alice@company.com");
        assert_eq!(user.name, "Alice Dupont");
        assert_eq!(
            user.picture.as_deref(),
            Some("https://example.com/photo.jpg")
        );
    }

    #[test]
    fn test_oidc_userinfo_without_picture() {
        let json = r#"{
            "sub": "1234567890",
            "email": "bob@company.com",
            "name": "Bob"
        }"#;

        let raw: RawOidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        let user = raw.normalize().expect("should normalize");
        assert_eq!(user.external_id, "1234567890");
        assert_eq!(user.name, "Bob");
        assert!(user.picture.is_none());
    }

    #[test]
    fn test_oidc_userinfo_cognito_format() {
        // Cognito returns username instead of name, no picture
        let json = r#"{
            "sub": "abc-def-123",
            "email": "user@example.com",
            "email_verified": "true",
            "username": "user@example.com"
        }"#;

        let raw: RawOidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        let user = raw.normalize().expect("should normalize");
        assert_eq!(user.external_id, "abc-def-123");
        assert_eq!(user.email, "user@example.com");
        assert_eq!(user.name, "user@example.com"); // Falls back to username
        assert!(user.picture.is_none());
    }

    #[test]
    fn test_oidc_userinfo_given_family_name() {
        let json = r#"{
            "sub": "xyz",
            "email": "jane@corp.com",
            "given_name": "Jane",
            "family_name": "Doe"
        }"#;

        let raw: RawOidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        let user = raw.normalize().expect("should normalize");
        assert_eq!(user.name, "Jane Doe");
    }

    #[test]
    fn test_oidc_userinfo_email_fallback_for_name() {
        // No name-related claims at all — falls back to email local part
        let json = r#"{
            "sub": "999",
            "email": "anon@example.com"
        }"#;

        let raw: RawOidcUserInfo = serde_json::from_str(json).expect("should deserialize");
        let user = raw.normalize().expect("should normalize");
        assert_eq!(user.name, "anon");
    }

    #[test]
    fn test_oidc_auth_url_with_redirect() {
        let config = test_auth_config();
        let client = OidcClient::from_legacy_google(&config).unwrap();

        let custom_redirect = "https://ffs.dev/auth/callback";
        let url = client.auth_url_with_redirect(custom_redirect);

        assert!(url.starts_with(GOOGLE_AUTH_URL));
        assert!(url.contains(&format!(
            "redirect_uri={}",
            urlencoding::encode(custom_redirect)
        )));
        // Original auth_url() still uses the config redirect_uri
        let default_url = client.auth_url();
        assert!(
            default_url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fauth%2Fcallback")
        );
    }

    #[test]
    fn test_oidc_auth_url_google_has_offline_access() {
        let config = test_auth_config();
        let client = OidcClient::from_legacy_google(&config).unwrap();
        let url = client.auth_url();

        // Google-specific params should be present
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=consent"));
    }

    #[test]
    fn test_oidc_auth_url_non_google_omits_google_params() {
        let config = OidcConfig {
            provider_key: Some("cognito".to_string()),
            provider_name: "Cognito".to_string(),
            client_id: "cognito-id".to_string(),
            client_secret: "cognito-secret".to_string(),
            redirect_uri: "http://localhost:3000/auth/callback".to_string(),
            auth_endpoint: Some(
                "https://mypool.auth.eu-west-1.amazoncognito.com/oauth2/authorize".to_string(),
            ),
            token_endpoint: Some(
                "https://mypool.auth.eu-west-1.amazoncognito.com/oauth2/token".to_string(),
            ),
            userinfo_endpoint: Some(
                "https://mypool.auth.eu-west-1.amazoncognito.com/oauth2/userInfo".to_string(),
            ),
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: Default::default(),
        };

        let client = OidcClient::from_config(&config).unwrap();
        let url = client.auth_url();

        // Google-specific params should NOT be present for Cognito
        assert!(
            !url.contains("access_type=offline"),
            "Cognito URL should not contain access_type=offline"
        );
        assert!(
            !url.contains("prompt=consent"),
            "Cognito URL should not contain prompt=consent"
        );
        // Standard OIDC params should be present
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=cognito-id"));
        assert!(url.contains("scope=openid"));
    }

    #[test]
    fn test_oidc_provider_key_from_explicit_config_google() {
        // When provider_key is explicitly "google" in OidcConfig,
        // Google-specific params should be auto-injected
        let config = OidcConfig {
            provider_key: Some("google".to_string()),
            provider_name: "Google (custom proxy)".to_string(),
            client_id: "proxy-id".to_string(),
            client_secret: "proxy-secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some("https://auth-proxy.internal/google/authorize".to_string()),
            token_endpoint: Some("https://auth-proxy.internal/google/token".to_string()),
            userinfo_endpoint: Some("https://auth-proxy.internal/google/userinfo".to_string()),
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: Default::default(),
        };

        let client = OidcClient::from_config(&config).unwrap();
        assert_eq!(client.provider_key, "google");
        assert!(client.is_google_provider());

        let url = client.auth_url();
        // Even though the URL is a proxy, provider_key="google" triggers Google params
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=consent"));
    }

    #[test]
    fn test_oidc_provider_key_fallback_slugify() {
        // When provider_key is None, it should be derived from provider_name
        let config = OidcConfig {
            provider_key: None,
            provider_name: "Google".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some("https://custom-google.example.com/auth".to_string()),
            token_endpoint: Some("https://custom-google.example.com/token".to_string()),
            userinfo_endpoint: None,
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: Default::default(),
        };

        let client = OidcClient::from_config(&config).unwrap();
        // Slugified "Google" → "google"
        assert_eq!(client.provider_key, "google");
        assert!(client.is_google_provider());
    }

    #[test]
    fn test_oidc_extra_auth_params_from_config() {
        // Custom extra_auth_params should be appended to the URL
        let mut params = std::collections::HashMap::new();
        params.insert("hd".to_string(), "example.com".to_string());
        params.insert("login_hint".to_string(), "user@example.com".to_string());

        let config = OidcConfig {
            provider_key: Some("google".to_string()),
            provider_name: "Google".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some(GOOGLE_AUTH_URL.to_string()),
            token_endpoint: Some(GOOGLE_TOKEN_URL.to_string()),
            userinfo_endpoint: None,
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: params,
        };

        let client = OidcClient::from_config(&config).unwrap();
        let url = client.auth_url();

        // Custom params present
        assert!(url.contains("hd=example.com"));
        assert!(url.contains("login_hint=user%40example.com"));
        // Google defaults also present (not overridden)
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=consent"));
    }

    #[test]
    fn test_oidc_extra_auth_params_override_google_defaults() {
        // User can override Google defaults via extra_auth_params
        let mut params = std::collections::HashMap::new();
        params.insert("prompt".to_string(), "select_account".to_string());

        let config = OidcConfig {
            provider_key: Some("google".to_string()),
            provider_name: "Google".to_string(),
            client_id: "id".to_string(),
            client_secret: "secret".to_string(),
            redirect_uri: "http://localhost/callback".to_string(),
            auth_endpoint: Some(GOOGLE_AUTH_URL.to_string()),
            token_endpoint: Some(GOOGLE_TOKEN_URL.to_string()),
            userinfo_endpoint: None,
            scopes: "openid email profile".to_string(),
            discovery_url: None,
            extra_auth_params: params,
        };

        let client = OidcClient::from_config(&config).unwrap();
        let url = client.auth_url();

        // Custom override should be used
        assert!(url.contains("prompt=select_account"));
        // access_type still gets the Google default (not overridden)
        assert!(url.contains("access_type=offline"));
        // "prompt=consent" should NOT be present (overridden)
        assert!(!url.contains("prompt=consent"));
    }

    #[test]
    fn test_oidc_userinfo_parse_error_no_pii_leak() {
        // Simulate a body that contains PII but is not valid RawOidcUserInfo
        // (e.g., missing required "sub" field)
        let pii_body = r#"{
            "email": "secret-user@company.com",
            "name": "John Secret",
            "picture": "https://example.com/secret-photo.jpg"
        }"#;

        let result: Result<RawOidcUserInfo, _> = serde_json::from_str(pii_body);
        assert!(result.is_err(), "Should fail without 'sub' claim");

        // Build the error message the same way exchange_code_with_redirect does
        let err = result.unwrap_err();
        let error_message = format!(
            "Failed to parse OIDC userinfo response (provider: {}, body_len: {}, content_type: {}): {}",
            "TestProvider", pii_body.len(), "application/json", err
        );

        // Verify no PII leaks in the error message
        assert!(
            !error_message.contains("secret-user@company.com"),
            "Error message must not contain email PII"
        );
        assert!(
            !error_message.contains("John Secret"),
            "Error message must not contain name PII"
        );
        assert!(
            !error_message.contains("secret-photo"),
            "Error message must not contain picture URL PII"
        );
        // Verify safe metadata IS present
        assert!(error_message.contains("TestProvider"));
        assert!(error_message.contains("body_len:"));
        assert!(error_message.contains("content_type:"));
    }

    #[test]
    fn test_oidc_userinfo_parse_error_invalid_json_no_pii_leak() {
        // Non-JSON body (e.g., HTML error page) — should not leak content either
        let html_body = "<html><body>Error: user secret-user@evil.com not found</body></html>";

        let result: Result<RawOidcUserInfo, _> = serde_json::from_str(html_body);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let error_message = format!(
            "Failed to parse OIDC userinfo response (provider: {}, body_len: {}, content_type: {}): {}",
            "TestProvider", html_body.len(), "text/html", err
        );

        // The serde error itself should not contain the full body
        assert!(
            !error_message.contains("secret-user@evil.com"),
            "Error message must not contain email from HTML body"
        );
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
        assert_eq!(doc.token_endpoint, "https://oauth2.googleapis.com/token");
        assert_eq!(
            doc.userinfo_endpoint.as_deref(),
            Some("https://openidconnect.googleapis.com/v1/userinfo")
        );
    }
}
