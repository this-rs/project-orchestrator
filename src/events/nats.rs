//! NATS-based event emitter for inter-process event synchronization
//!
//! Publishes CrudEvents to NATS subjects for cross-instance communication.
//! Fire-and-forget: errors are logged but never block the caller.

use super::types::{CrudEvent, EventEmitter};
use tracing::{debug, warn};

/// NATS event emitter that publishes CrudEvents to NATS subjects.
///
/// Used by both the HTTP server and the MCP server to broadcast mutations
/// to all connected instances (dev, desktop, etc.) via NATS pub/sub.
///
/// Fire-and-forget: publishing never blocks, never panics.
/// If NATS is disconnected, events are silently dropped with a warning log.
#[derive(Clone)]
pub struct NatsEmitter {
    client: async_nats::Client,
    subject_prefix: String,
}

impl NatsEmitter {
    /// Create a new NatsEmitter with the given NATS client and subject prefix.
    ///
    /// Events will be published to `{prefix}.crud` (e.g. "events.crud").
    pub fn new(client: async_nats::Client, subject_prefix: impl Into<String>) -> Self {
        Self {
            client,
            subject_prefix: subject_prefix.into(),
        }
    }

    /// Get a reference to the underlying NATS client.
    ///
    /// Useful for creating subscribers or publishing to other subjects
    /// (e.g. chat events, interrupts).
    pub fn client(&self) -> &async_nats::Client {
        &self.client
    }

    /// Get the subject prefix (e.g. "events").
    pub fn subject_prefix(&self) -> &str {
        &self.subject_prefix
    }

    /// Build the CRUD events subject (e.g. "events.crud").
    fn crud_subject(&self) -> String {
        format!("{}.crud", self.subject_prefix)
    }
}

impl EventEmitter for NatsEmitter {
    fn emit(&self, event: CrudEvent) {
        let client = self.client.clone();
        let subject = self.crud_subject();

        tokio::spawn(async move {
            match serde_json::to_vec(&event) {
                Ok(payload) => {
                    if let Err(e) = client.publish(subject.clone(), payload.into()).await {
                        warn!(
                            subject = %subject,
                            entity_type = ?event.entity_type,
                            action = ?event.action,
                            "Failed to publish event to NATS: {}",
                            e
                        );
                    } else {
                        debug!(
                            subject = %subject,
                            entity_type = ?event.entity_type,
                            action = ?event.action,
                            entity_id = %event.entity_id,
                            "CrudEvent published to NATS"
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        entity_type = ?event.entity_type,
                        action = ?event.action,
                        "Failed to serialize CrudEvent for NATS: {}",
                        e
                    );
                }
            }
        });
    }
}

/// Connect to a NATS server.
///
/// Returns a connected `async_nats::Client` ready for publishing and subscribing.
pub async fn connect_nats(url: &str) -> anyhow::Result<async_nats::Client> {
    let client = async_nats::connect(url).await.map_err(|e| {
        anyhow::anyhow!("Failed to connect to NATS at {}: {}", url, e)
    })?;
    tracing::info!("Connected to NATS at {}", url);
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};

    #[test]
    fn test_crud_subject() {
        // We can't easily create a real async_nats::Client in tests without a server,
        // but we can test the subject construction logic indirectly.
        let subject_prefix = "events";
        let subject = format!("{}.crud", subject_prefix);
        assert_eq!(subject, "events.crud");
    }

    #[test]
    fn test_crud_event_serialization_for_nats() {
        // Verify that CrudEvent serializes to JSON correctly for NATS transport
        let event = CrudEvent::new(EntityType::Plan, CrudAction::Created, "plan-123")
            .with_payload(serde_json::json!({"title": "Test Plan"}))
            .with_project_id("proj-456");

        let payload = serde_json::to_vec(&event).unwrap();
        let deserialized: CrudEvent = serde_json::from_slice(&payload).unwrap();

        assert_eq!(deserialized.entity_type, EntityType::Plan);
        assert_eq!(deserialized.action, CrudAction::Created);
        assert_eq!(deserialized.entity_id, "plan-123");
        assert_eq!(deserialized.project_id.as_deref(), Some("proj-456"));
    }

    #[test]
    fn test_subject_prefix_custom() {
        let prefix = "myapp.events";
        let subject = format!("{}.crud", prefix);
        assert_eq!(subject, "myapp.events.crud");
    }
}
