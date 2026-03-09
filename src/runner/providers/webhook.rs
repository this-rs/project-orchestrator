//! WebhookProvider — external HTTP webhook receiver for trigger activation.
//!
//! Adds a POST /api/webhooks/:trigger_id endpoint that receives external
//! payloads (e.g., GitHub push events). Validates HMAC-SHA256 signatures
//! and filters by event type and branch pattern.

use super::TriggerProvider;
use crate::runner::models::TriggerType;
use anyhow::Result;
use async_trait::async_trait;
use tracing::info;

/// Webhook-based trigger provider.
///
/// The actual HTTP endpoint is registered in `api/routes.rs` as
/// `POST /api/webhooks/:trigger_id`. This provider just marks itself
/// as the Webhook type — the route handler does the actual work.
///
/// Config format for Webhook triggers:
/// ```json
/// {
///   "secret": "hmac_secret_for_signature_validation",
///   "event_filter": ["push", "pull_request.closed"],
///   "branch_pattern": "main|release/.*"
/// }
/// ```
#[derive(Debug, Default)]
pub struct WebhookProvider;

impl WebhookProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TriggerProvider for WebhookProvider {
    async fn setup(&self) -> Result<()> {
        info!("WebhookProvider ready — endpoint: POST /api/webhooks/:trigger_id");
        Ok(())
    }

    async fn teardown(&self) -> Result<()> {
        info!("WebhookProvider teardown complete");
        Ok(())
    }

    fn provider_type(&self) -> TriggerType {
        TriggerType::Webhook
    }
}

/// Validate an HMAC-SHA256 signature from a GitHub webhook.
///
/// The expected format is `sha256=<hex_signature>`.
pub fn validate_github_signature(secret: &str, body: &[u8], signature_header: &str) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let Some(hex_sig) = signature_header.strip_prefix("sha256=") else {
        return false;
    };

    let Ok(expected) = hex::decode(hex_sig) else {
        return false;
    };

    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret.as_bytes()) else {
        return false;
    };

    mac.update(body);
    mac.verify_slice(&expected).is_ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_github_signature_valid() {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let secret = "my_webhook_secret";
        let body = b"test payload";

        // Compute the expected signature
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        let result = mac.finalize();
        let hex_sig = hex::encode(result.into_bytes());
        let header = format!("sha256={}", hex_sig);

        assert!(validate_github_signature(secret, body, &header));
    }

    #[test]
    fn test_validate_github_signature_invalid() {
        assert!(!validate_github_signature(
            "secret",
            b"body",
            "sha256=0000000000000000000000000000000000000000000000000000000000000000"
        ));
    }

    #[test]
    fn test_validate_github_signature_bad_format() {
        assert!(!validate_github_signature("secret", b"body", "md5=abc"));
        assert!(!validate_github_signature(
            "secret",
            b"body",
            "sha256=notahex"
        ));
    }

    #[tokio::test]
    async fn test_webhook_provider_type() {
        let provider = WebhookProvider::new();
        assert_eq!(provider.provider_type(), TriggerType::Webhook);
    }
}
