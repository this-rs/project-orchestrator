//! NATS-based event emitter for inter-process event synchronization
//!
//! Publishes CrudEvents to NATS subjects for cross-instance communication.
//! Also handles chat event pub/sub and interrupt signaling via NATS.
//!
//! Fire-and-forget publishing: errors are logged but never block the caller.

use super::types::{CrudEvent, EventEmitter};
use crate::chat::types::ChatEvent;
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

    // ========================================================================
    // Chat event pub/sub
    // ========================================================================

    /// Build the chat events subject for a session (e.g. "events.chat.{session_id}").
    pub fn chat_subject(&self, session_id: &str) -> String {
        format!("{}.chat.{}", self.subject_prefix, session_id)
    }

    /// Build the interrupt subject for a session (e.g. "events.chat.{session_id}.interrupt").
    pub fn interrupt_subject(&self, session_id: &str) -> String {
        format!("{}.chat.{}.interrupt", self.subject_prefix, session_id)
    }

    /// Publish a ChatEvent to the session's NATS subject.
    ///
    /// Fire-and-forget: errors are logged but never block the caller.
    pub fn publish_chat_event(&self, session_id: &str, event: ChatEvent) {
        let client = self.client.clone();
        let subject = self.chat_subject(session_id);

        tokio::spawn(async move {
            match serde_json::to_vec(&event) {
                Ok(payload) => {
                    if let Err(e) = client.publish(subject.clone(), payload.into()).await {
                        warn!(
                            subject = %subject,
                            event_type = %event.event_type(),
                            "Failed to publish ChatEvent to NATS: {}",
                            e
                        );
                    } else {
                        debug!(
                            subject = %subject,
                            event_type = %event.event_type(),
                            "ChatEvent published to NATS"
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        event_type = %event.event_type(),
                        "Failed to serialize ChatEvent for NATS: {}",
                        e
                    );
                }
            }
        });
    }

    /// Publish an interrupt signal for a chat session.
    ///
    /// The payload is a simple JSON `{"interrupt": true}` message.
    /// Fire-and-forget: errors are logged but never block the caller.
    pub fn publish_interrupt(&self, session_id: &str) {
        let client = self.client.clone();
        let subject = self.interrupt_subject(session_id);

        tokio::spawn(async move {
            let payload = b"{\"interrupt\":true}";
            if let Err(e) = client
                .publish(subject.clone(), payload.to_vec().into())
                .await
            {
                warn!(
                    subject = %subject,
                    "Failed to publish interrupt to NATS: {}",
                    e
                );
            } else {
                debug!(
                    subject = %subject,
                    "Interrupt published to NATS"
                );
            }
        });
    }

    /// Subscribe to chat events for a session from NATS.
    ///
    /// Returns a NATS subscriber that yields messages on `events.chat.{session_id}`.
    /// Each message payload is a JSON-serialized `ChatEvent`.
    pub async fn subscribe_chat_events(
        &self,
        session_id: &str,
    ) -> anyhow::Result<async_nats::Subscriber> {
        let subject = self.chat_subject(session_id);
        let subscriber = self.client.subscribe(subject.clone()).await.map_err(|e| {
            anyhow::anyhow!("Failed to subscribe to NATS chat events {}: {}", subject, e)
        })?;
        debug!(subject = %subject, "Subscribed to NATS chat events");
        Ok(subscriber)
    }

    /// Subscribe to interrupt signals for a session from NATS.
    ///
    /// Returns a NATS subscriber that yields messages on `events.chat.{session_id}.interrupt`.
    pub async fn subscribe_interrupt(
        &self,
        session_id: &str,
    ) -> anyhow::Result<async_nats::Subscriber> {
        let subject = self.interrupt_subject(session_id);
        let subscriber = self.client.subscribe(subject.clone()).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to subscribe to NATS interrupt {}: {}",
                subject,
                e
            )
        })?;
        debug!(subject = %subject, "Subscribed to NATS interrupt");
        Ok(subscriber)
    }

    /// Subscribe to CRUD events from NATS.
    ///
    /// Returns a NATS subscriber that yields messages on `events.crud`.
    /// Each message payload is a JSON-serialized `CrudEvent`.
    pub async fn subscribe_crud_events(&self) -> anyhow::Result<async_nats::Subscriber> {
        let subject = self.crud_subject();
        let subscriber = self.client.subscribe(subject.clone()).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to subscribe to NATS CRUD events {}: {}",
                subject,
                e
            )
        })?;
        debug!(subject = %subject, "Subscribed to NATS CRUD events");
        Ok(subscriber)
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

    // ========================================================================
    // Subject construction
    // ========================================================================

    #[test]
    fn test_crud_subject() {
        let subject_prefix = "events";
        let subject = format!("{}.crud", subject_prefix);
        assert_eq!(subject, "events.crud");
    }

    #[test]
    fn test_subject_prefix_custom() {
        let prefix = "myapp.events";
        let subject = format!("{}.crud", prefix);
        assert_eq!(subject, "myapp.events.crud");
    }

    #[test]
    fn test_chat_subject() {
        let prefix = "events";
        let session_id = "abc-123";
        let subject = format!("{}.chat.{}", prefix, session_id);
        assert_eq!(subject, "events.chat.abc-123");
    }

    #[test]
    fn test_interrupt_subject() {
        let prefix = "events";
        let session_id = "abc-123";
        let subject = format!("{}.chat.{}.interrupt", prefix, session_id);
        assert_eq!(subject, "events.chat.abc-123.interrupt");
    }

    #[test]
    fn test_chat_subject_with_custom_prefix() {
        let prefix = "po.dev";
        let session_id = "sess-456";
        let subject = format!("{}.chat.{}", prefix, session_id);
        assert_eq!(subject, "po.dev.chat.sess-456");

        let interrupt = format!("{}.chat.{}.interrupt", prefix, session_id);
        assert_eq!(interrupt, "po.dev.chat.sess-456.interrupt");
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    #[test]
    fn test_crud_event_serialization_for_nats() {
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
    fn test_chat_event_serialization_for_nats() {
        // Verify all ChatEvent variants serialize/deserialize correctly for NATS
        let events = vec![
            ChatEvent::UserMessage {
                content: "Hello".into(),
            },
            ChatEvent::AssistantText {
                content: "Hi there!".into(),
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
            },
            ChatEvent::ToolUse {
                id: "tu_1".into(),
                tool: "create_plan".into(),
                input: serde_json::json!({"title": "Plan"}),
            },
            ChatEvent::ToolResult {
                id: "tu_1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
            },
            ChatEvent::ToolUseInputResolved {
                id: "tu_1".into(),
                input: serde_json::json!({"title": "Resolved"}),
            },
            ChatEvent::PermissionRequest {
                id: "pr_1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            },
            ChatEvent::InputRequest {
                prompt: "Choose:".into(),
                options: Some(vec!["A".into(), "B".into()]),
            },
            ChatEvent::Result {
                session_id: "sess-1".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
            },
            ChatEvent::StreamDelta {
                text: "tok".into(),
            },
            ChatEvent::StreamingStatus { is_streaming: true },
            ChatEvent::Error {
                message: "fail".into(),
            },
        ];

        for event in &events {
            let payload = serde_json::to_vec(event).unwrap();
            let deserialized: ChatEvent = serde_json::from_slice(&payload).unwrap();
            assert_eq!(event.event_type(), deserialized.event_type());
        }
    }

    #[test]
    fn test_interrupt_payload() {
        // Verify the interrupt payload is valid JSON
        let payload = b"{\"interrupt\":true}";
        let parsed: serde_json::Value = serde_json::from_slice(payload).unwrap();
        assert_eq!(parsed["interrupt"], true);
    }
}
