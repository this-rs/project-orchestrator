//! Refresh token generation, hashing, and cookie helpers.
//!
//! The refresh token is an opaque 256-bit random value encoded as hex (64 chars).
//! It is stored **hashed** (SHA-256) in the database — the raw token only exists
//! in the Set-Cookie header sent to the client. This means:
//! - Even a full DB dump doesn't reveal usable tokens
//! - Token comparison is done by hashing the incoming cookie and looking up the hash
//!
//! Cookie format: `refresh_token=<hex>; HttpOnly; SameSite=Lax; Path=/; [Secure]`
//! - `HttpOnly`: not accessible via JavaScript (XSS protection)
//! - `SameSite=Lax`: not sent on cross-site POST (CSRF protection)
//! - `Path=/`: sent for all routes including `/ws/*` upgrades
//! - `Secure`: only over HTTPS (omitted in localhost/non-TLS mode)

use axum::http::HeaderValue;
use sha2::{Digest, Sha256};

/// Cookie name for the refresh token.
pub const REFRESH_COOKIE_NAME: &str = "refresh_token";

/// Generate a cryptographically random 256-bit token encoded as hex (64 chars).
pub fn generate_token() -> String {
    let bytes: [u8; 32] = rand::random();
    hex::encode(bytes)
}

/// Hash a raw token with SHA-256 and return the hex digest.
///
/// This is stored in the database — the raw token is never persisted.
pub fn hash_token(raw_token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw_token.as_bytes());
    hex::encode(hasher.finalize())
}

/// Build the `Set-Cookie` header value for the refresh token.
///
/// # Arguments
/// * `raw_token` - The raw (unhashed) token to put in the cookie
/// * `max_age_secs` - Cookie lifetime in seconds (matches refresh token expiry)
/// * `is_secure` - Whether to add the `Secure` flag (true for HTTPS, false for localhost)
pub fn build_refresh_cookie(raw_token: &str, max_age_secs: u64, is_secure: bool) -> HeaderValue {
    let secure_flag = if is_secure { "; Secure" } else { "" };
    let cookie = format!(
        "{}={}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}{}",
        REFRESH_COOKIE_NAME, raw_token, max_age_secs, secure_flag
    );
    // HeaderValue::from_str can fail on non-ASCII, but our token is hex-only
    HeaderValue::from_str(&cookie).expect("cookie value is valid ASCII")
}

/// Build a `Set-Cookie` header that clears (deletes) the refresh token cookie.
///
/// Used by `POST /auth/logout` to remove the cookie from the browser.
pub fn build_clear_cookie(is_secure: bool) -> HeaderValue {
    let secure_flag = if is_secure { "; Secure" } else { "" };
    let cookie = format!(
        "{}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0{}",
        REFRESH_COOKIE_NAME, secure_flag
    );
    HeaderValue::from_str(&cookie).expect("cookie value is valid ASCII")
}

/// Extract the refresh token from a `Cookie` header value.
///
/// Parses the cookie string to find `refresh_token=<value>`.
pub fn extract_refresh_token_from_cookie(cookie_header: &str) -> Option<String> {
    for part in cookie_header.split(';') {
        let trimmed = part.trim();
        if let Some(value) = trimmed.strip_prefix(&format!("{}=", REFRESH_COOKIE_NAME)) {
            let token = value.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// Determine whether the `Secure` flag should be set on cookies.
///
/// Returns `true` if the server is accessed via HTTPS (public_url starts with https://),
/// `false` for localhost / non-TLS setups (e.g. Tauri desktop).
pub fn should_set_secure(public_url: Option<&str>) -> bool {
    public_url.is_some_and(|url| url.starts_with("https://"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_token_length_and_uniqueness() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_eq!(t1.len(), 64, "Token should be 64 hex chars (256 bits)");
        assert_ne!(t1, t2, "Two tokens should be different");
    }

    #[test]
    fn test_hash_token_deterministic() {
        let token = "abc123";
        let h1 = hash_token(token);
        let h2 = hash_token(token);
        assert_eq!(h1, h2, "Same input should produce same hash");
        assert_eq!(h1.len(), 64, "SHA-256 hex digest is 64 chars");
    }

    #[test]
    fn test_hash_token_different_input() {
        let h1 = hash_token("token_a");
        let h2 = hash_token("token_b");
        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_build_refresh_cookie_no_secure() {
        let cookie = build_refresh_cookie("mytoken123", 604800, false);
        let s = cookie.to_str().unwrap();
        assert!(s.contains("refresh_token=mytoken123"));
        assert!(s.contains("HttpOnly"));
        assert!(s.contains("SameSite=Lax"));
        assert!(s.contains("Path=/"));
        assert!(s.contains("Max-Age=604800"));
        assert!(!s.contains("Secure"), "No Secure flag in non-TLS mode");
    }

    #[test]
    fn test_build_refresh_cookie_with_secure() {
        let cookie = build_refresh_cookie("mytoken123", 604800, true);
        let s = cookie.to_str().unwrap();
        assert!(s.contains("Secure"), "Secure flag required in TLS mode");
    }

    #[test]
    fn test_build_clear_cookie() {
        let cookie = build_clear_cookie(false);
        let s = cookie.to_str().unwrap();
        assert!(s.contains("refresh_token=;"));
        assert!(s.contains("Max-Age=0"));
        assert!(s.contains("HttpOnly"));
        assert!(s.contains("Path=/"));
    }

    #[test]
    fn test_extract_refresh_token_from_cookie() {
        // Single cookie
        let token = extract_refresh_token_from_cookie("refresh_token=abc123");
        assert_eq!(token, Some("abc123".to_string()));

        // Multiple cookies
        let token =
            extract_refresh_token_from_cookie("session=xyz; refresh_token=def456; other=val");
        assert_eq!(token, Some("def456".to_string()));

        // No refresh_token
        let token = extract_refresh_token_from_cookie("session=xyz; other=val");
        assert_eq!(token, None);

        // Empty value
        let token = extract_refresh_token_from_cookie("refresh_token=");
        assert_eq!(token, None);
    }

    #[test]
    fn test_should_set_secure() {
        assert!(should_set_secure(Some("https://ffs.dev")));
        assert!(!should_set_secure(Some("http://localhost:8080")));
        assert!(!should_set_secure(None));
    }
}
