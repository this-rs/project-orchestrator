//! HTTP-based event notifier for bridging MCP → HTTP server
//!
//! **DEPRECATED**: Use `NatsEmitter` instead for inter-process event sync.
//! This module is kept for backward compatibility with existing deployments
//! that haven't migrated to NATS yet. It will be removed in a future version.
//!
//! Sends CrudEvents to the HTTP server via POST /internal/events.
//! Fire-and-forget: errors are logged but never block the caller.

use super::types::{CrudEvent, EventEmitter};
use std::time::Duration;
use tracing::warn;

/// HTTP client that forwards CrudEvents to the HTTP server's /internal/events endpoint.
///
/// **DEPRECATED**: Use `NatsEmitter` instead for inter-process event sync.
/// This type is kept for backward compatibility with deployments using `MCP_HTTP_URL`.
/// Migrate to `NATS_URL` for reliable cross-instance communication.
#[deprecated(
    since = "0.2.0",
    note = "Use NatsEmitter with NATS_URL instead. EventNotifier will be removed in a future version."
)]
#[derive(Clone)]
pub struct EventNotifier {
    client: reqwest::Client,
    url: String,
}

#[allow(deprecated)]
impl EventNotifier {
    /// Create a new EventNotifier targeting the given base URL
    ///
    /// The base_url should be the HTTP server root (e.g. "http://localhost:8080").
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to create reqwest client");

        let url = format!("{}/internal/events", base_url.trim_end_matches('/'));

        Self { client, url }
    }
}

#[allow(deprecated)]
impl EventEmitter for EventNotifier {
    fn emit(&self, event: CrudEvent) {
        let client = self.client.clone();
        let url = self.url.clone();

        tokio::spawn(async move {
            if let Err(e) = client.post(&url).json(&event).send().await {
                warn!(
                    url = %url,
                    entity_type = ?event.entity_type,
                    action = ?event.action,
                    "Failed to forward event to HTTP server: {}",
                    e
                );
            }
        });
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::events::EntityType;

    #[test]
    fn test_new_builds_correct_url() {
        let notifier = EventNotifier::new("http://localhost:8080");
        assert_eq!(notifier.url, "http://localhost:8080/internal/events");
    }

    #[test]
    fn test_new_trims_trailing_slash() {
        let notifier = EventNotifier::new("http://localhost:8080/");
        assert_eq!(notifier.url, "http://localhost:8080/internal/events");
    }

    #[test]
    fn test_clone() {
        let notifier = EventNotifier::new("http://localhost:8080");
        let cloned = notifier.clone();
        assert_eq!(cloned.url, notifier.url);
    }

    #[tokio::test]
    async fn test_emit_fire_and_forget_no_panic() {
        // Even with no server listening, emit should not panic
        let notifier = EventNotifier::new("http://127.0.0.1:1"); // Port 1 — nothing listening
        notifier.emit_created(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"title": "Test"}),
            None,
        );
        // Give the spawned task a moment to execute (and fail gracefully)
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_emit_deleted_no_panic() {
        let notifier = EventNotifier::new("http://127.0.0.1:1");
        notifier.emit_deleted(EntityType::Task, "task-1", Some("proj-1".into()));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_emit_linked_no_panic() {
        let notifier = EventNotifier::new("http://127.0.0.1:1");
        notifier.emit_linked(
            EntityType::Task,
            "task-1",
            EntityType::Release,
            "release-1",
            None,
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_emit_unlinked_no_panic() {
        let notifier = EventNotifier::new("http://127.0.0.1:1");
        notifier.emit_unlinked(
            EntityType::Note,
            "note-1",
            EntityType::Task,
            "task-2",
            Some("proj-1".into()),
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_emit_updated_no_panic() {
        let notifier = EventNotifier::new("http://127.0.0.1:1");
        notifier.emit_updated(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"status": "completed"}),
            None,
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
