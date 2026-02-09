//! Google OAuth2 Authorization Code Flow
//!
//! Handles the server-side of Google OAuth:
//! 1. Generate the authorization URL (user redirects to Google)
//! 2. Exchange the authorization code for an access token
//! 3. Fetch user info from Google's userinfo endpoint

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::AuthConfig;

/// User information retrieved from Google after successful OAuth
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleUserInfo {
    /// Google's unique user identifier (the "sub" claim)
    #[serde(rename = "sub")]
    pub google_id: String,
    /// User's email address
    pub email: String,
    /// User's display name
    pub name: String,
    /// URL to the user's profile picture
    pub picture: Option<String>,
}

/// Google OAuth2 client for authorization code flow
pub struct GoogleOAuthClient {
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    http_client: reqwest::Client,
}

/// Google token endpoint response
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[allow(dead_code)]
    token_type: String,
    #[allow(dead_code)]
    expires_in: Option<u64>,
}

const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v3/userinfo";

impl GoogleOAuthClient {
    /// Create a new Google OAuth client from the auth configuration.
    pub fn new(config: &AuthConfig) -> Self {
        Self {
            client_id: config.google_client_id.clone(),
            client_secret: config.google_client_secret.clone(),
            redirect_uri: config.google_redirect_uri.clone(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Generate the Google OAuth authorization URL.
    ///
    /// The user should be redirected to this URL to initiate the OAuth flow.
    /// After consent, Google redirects back to `redirect_uri` with a `code` parameter.
    pub fn auth_url(&self) -> String {
        format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
            GOOGLE_AUTH_URL,
            urlencoding::encode(&self.client_id),
            urlencoding::encode(&self.redirect_uri),
            urlencoding::encode("openid email profile"),
        )
    }

    /// Exchange an authorization code for user information.
    ///
    /// This performs two steps:
    /// 1. POST to Google's token endpoint to exchange the code for an access token
    /// 2. GET Google's userinfo endpoint with the access token to retrieve user details
    pub async fn exchange_code(&self, code: &str) -> Result<GoogleUserInfo> {
        // Step 1: Exchange code for access token
        let token_response = self
            .http_client
            .post(GOOGLE_TOKEN_URL)
            .form(&[
                ("code", code),
                ("client_id", &self.client_id),
                ("client_secret", &self.client_secret),
                ("redirect_uri", &self.redirect_uri),
                ("grant_type", "authorization_code"),
            ])
            .send()
            .await
            .context("Failed to request Google token")?;

        if !token_response.status().is_success() {
            let status = token_response.status();
            let body = token_response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            bail!("Google token exchange failed ({}): {}", status, body);
        }

        let token: TokenResponse = token_response
            .json()
            .await
            .context("Failed to parse Google token response")?;

        // Step 2: Fetch user info with access token
        let userinfo_response = self
            .http_client
            .get(GOOGLE_USERINFO_URL)
            .bearer_auth(&token.access_token)
            .send()
            .await
            .context("Failed to request Google userinfo")?;

        if !userinfo_response.status().is_success() {
            let status = userinfo_response.status();
            let body = userinfo_response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            bail!("Google userinfo fetch failed ({}): {}", status, body);
        }

        let user_info: GoogleUserInfo = userinfo_response
            .json()
            .await
            .context("Failed to parse Google userinfo response")?;

        Ok(user_info)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_auth_config() -> AuthConfig {
        AuthConfig {
            google_client_id: "123456.apps.googleusercontent.com".to_string(),
            google_client_secret: "secret123".to_string(),
            google_redirect_uri: "http://localhost:3000/auth/callback".to_string(),
            jwt_secret: "test-secret-key-minimum-32-chars!!".to_string(),
            jwt_expiry_secs: 28800,
            allowed_email_domain: Some("ffs.holdings".to_string()),
            frontend_url: Some("http://localhost:3000".to_string()),
        }
    }

    #[test]
    fn test_auth_url_contains_required_params() {
        let client = GoogleOAuthClient::new(&test_auth_config());
        let url = client.auth_url();

        assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth"));
        assert!(url.contains("client_id=123456.apps.googleusercontent.com"));
        assert!(
            url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fauth%2Fcallback")
        );
        assert!(url.contains("response_type=code"));
        assert!(url.contains("scope=openid"));
        assert!(url.contains("email"));
        assert!(url.contains("profile"));
        assert!(url.contains("access_type=offline"));
    }

    #[test]
    fn test_google_oauth_client_construction() {
        let config = test_auth_config();
        let client = GoogleOAuthClient::new(&config);

        assert_eq!(client.client_id, "123456.apps.googleusercontent.com");
        assert_eq!(client.client_secret, "secret123");
        assert_eq!(
            client.redirect_uri,
            "http://localhost:3000/auth/callback"
        );
    }

    #[test]
    fn test_google_user_info_deserialization() {
        let json = r#"{
            "sub": "1234567890",
            "email": "alice@ffs.holdings",
            "name": "Alice Dupont",
            "picture": "https://lh3.googleusercontent.com/a/photo"
        }"#;

        let user: GoogleUserInfo = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(user.google_id, "1234567890");
        assert_eq!(user.email, "alice@ffs.holdings");
        assert_eq!(user.name, "Alice Dupont");
        assert_eq!(
            user.picture.as_deref(),
            Some("https://lh3.googleusercontent.com/a/photo")
        );
    }

    #[test]
    fn test_google_user_info_without_picture() {
        let json = r#"{
            "sub": "1234567890",
            "email": "bob@ffs.holdings",
            "name": "Bob"
        }"#;

        let user: GoogleUserInfo = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(user.google_id, "1234567890");
        assert!(user.picture.is_none());
    }
}
