//! Inter-instance chat transport, abstracted from the concrete bus.
//!
//! The CRUD [`EventEmitter`](super::types::EventEmitter) only models
//! fire-and-forget entity events. The chat domain needs more: per-session
//! publish/subscribe, interrupt and cancel-tools signalling, streaming
//! snapshots and a request/reply "cancel a task" relay. Today all of it is
//! consumed concretely as `Arc<NatsEmitter>`; this module introduces the
//! [`ChatTransport`] trait so a single-instance PO (no NATS) can run on an
//! in-memory transport and the NATS-backed path is one adapter among others.
//!
//! This is a **first increment**: only `publish_chat_event` /
//! `subscribe_chat_events` are migrated onto the trait. The other operations
//! are declared here and implemented by both transports, but their call sites
//! still go through `NatsEmitter` until the next slice.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::nats::ChatRpcResponse;
use super::transport::{Message, Subject, Subscriber};
use super::types::TransportState;
use crate::chat::types::ChatEvent;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Why a cancel asked of another instance came back without an answer
/// ([`ChatTransport::request_cancel`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayFailure {
    /// Nobody subscribes to the subject: no instance holds the session live.
    NoResponders,
    /// No answer within the timeout: the owner is there but did not answer.
    TimedOut,
    /// The request could not be sent or read.
    Transport(String),
}

/// Snapshot of a streaming session's current state, used for cross-instance mid-stream join.
///
/// When a client connects to a remote instance (one that doesn't own the streaming session),
/// this struct is returned via request/reply so the client can reconstruct the
/// in-progress assistant turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSnapshot {
    pub is_streaming: bool,
    pub partial_text: String,
    pub events: Vec<ChatEvent>,
}

/// Inter-instance transport for chat: publish/subscribe of a session's events
/// plus the cross-instance control signals.
///
/// # Guarantees
///
/// * **Delivery** — `publish_*` methods are fire-and-forget: they return
///   immediately and a failure is logged, never raised to the caller. A
///   `subscribe_*` that returns `Ok` yields every message published *after*
///   the subscription was established (no replay of what came before).
/// * **Order** — within one session, events are delivered in publication
///   order. The NATS adapter preserves the per-connection order; the
///   in-memory transport is a FIFO broadcast per session.
/// * **Timeout** — `request_streaming_snapshot` waits 2 s, `request_cancel`
///   waits the transport's cancel timeout (10 s by default), then reports
///   `None` / [`RelayFailure::TimedOut`].
/// * **Mono-instance losses** — with no second instance, an interrupt, a
///   cancel-tools signal or a streaming-snapshot request has no remote
///   listener: the in-memory transport reports them as no-ops (an empty
///   subscription / `None` / [`RelayFailure::NoResponders`]). Nothing is
///   lost locally: the local broadcast (`events_tx`) still carries the event.
#[async_trait::async_trait]
pub trait ChatTransport: Send + Sync {
    /// Publish a [`ChatEvent`] to the session's channel. Fire-and-forget.
    fn publish_chat_event(&self, session_id: &str, event: ChatEvent);

    /// Subscribe to a session's chat events.
    async fn subscribe_chat_events(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish an interrupt signal for a session. Fire-and-forget.
    fn publish_chat_interrupt(&self, session_id: &str);

    /// Subscribe to a session's interrupt signals.
    async fn subscribe_chat_interrupt(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish a cancel-tools signal for a session. Fire-and-forget.
    fn publish_cancel_tools(&self, session_id: &str);

    /// Subscribe to a session's cancel-tools signals.
    async fn subscribe_cancel_tools(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Request a streaming snapshot from the instance that owns the session.
    async fn request_streaming_snapshot(&self, session_id: &str) -> Option<StreamingSnapshot>;

    /// Subscribe to streaming-snapshot requests (the owning instance answers each).
    async fn subscribe_snapshot_requests(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Publish an opaque payload on an opaque reply subject (the reply inbox of
    /// a request this instance received). Used by the request/reply relays
    /// (snapshots, RPC, `cancel_relay`) to answer an inbound request.
    async fn publish(&self, subject: &Subject, payload: Vec<u8>) -> anyhow::Result<()>;

    /// Flush the transport so publishes already queued reach the far side.
    async fn flush(&self) -> anyhow::Result<()>;

    /// Health of the transport connection, for the status endpoint:
    /// connected, disconnected, or unknown (a transport with no remote).
    fn transport_state(&self) -> TransportState;

    /// Build the cancel-tools subject for a session.
    fn cancel_tools_subject(&self, session_id: &str) -> String;

    /// Build the cancel-task subject for a session.
    fn cancel_task_subject(&self, session_id: &str) -> String;

    /// Subscribe to cancel-task requests for a session (the owning instance
    /// answers each on its reply subject).
    async fn subscribe_cancel_task(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Subscribe to RPC "send a message" requests for a session (the owning
    /// instance processes each and answers with a `ChatRpcResponse`).
    async fn subscribe_rpc_send(&self, session_id: &str) -> anyhow::Result<Subscriber>;

    /// Ask the instance that owns a session to send a message to it (the RPC
    /// "send a message" request), returning its `ChatRpcResponse`, or `None`
    /// when nobody answers within the timeout.
    async fn request_send_message(
        &self,
        session_id: &str,
        message: &str,
        message_type: &str,
    ) -> Option<ChatRpcResponse>;

    /// Ask the instance that owns a session to cancel it, returning its raw
    /// answer or why there is none ([`RelayFailure`]).
    /// Not an `async fn`: the provisional-answer predicate is a higher-ranked
    /// `Fn(&[u8])`, which `#[async_trait]` would otherwise tie to the future's
    /// lifetime and break the delegation to the NATS emitter.
    fn request_cancel<'a>(
        &'a self,
        subject: String,
        payload: serde_json::Value,
        is_provisional: &'a (dyn Fn(&[u8]) -> bool + Sync),
        grace: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, RelayFailure>> + Send + 'a>,
    >;
}

/// In-memory [`ChatTransport`]: chat events stay in this process.
///
/// Lets a single-instance PO run without NATS. `publish_chat_event` broadcasts
/// to the process-local subscribers of the session; there are none in a single
/// instance, so the behaviour is a no-op exactly like the absent-NATS path.
/// Cross-instance control (`request_cancel`, snapshots) has no remote owner and
/// reports the documented mono-instance results.
#[derive(Clone, Default)]
pub struct InMemoryChatTransport {
    sessions: Arc<Mutex<HashMap<String, tokio::sync::broadcast::Sender<ChatEvent>>>>,
}

impl InMemoryChatTransport {
    /// A transport with no sessions yet.
    pub fn new() -> Self {
        Self::default()
    }

    fn sender(&self, session_id: &str) -> tokio::sync::broadcast::Sender<ChatEvent> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| tokio::sync::broadcast::channel(1024).0)
            .clone()
    }
}

#[async_trait::async_trait]
impl ChatTransport for InMemoryChatTransport {
    fn publish_chat_event(&self, session_id: &str, event: ChatEvent) {
        let sessions = self.sessions.lock().unwrap();
        if let Some(tx) = sessions.get(session_id) {
            // No receivers -> `Err`, which is fine: nobody is listening.
            let _ = tx.send(event);
        }
    }

    async fn subscribe_chat_events(&self, session_id: &str) -> anyhow::Result<Subscriber> {
        let rx = self.sender(session_id).subscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(rx)
            .filter_map(|r| async move { r.ok() })
            .map(|event: ChatEvent| Message {
                payload: serde_json::to_vec(&event).unwrap_or_default(),
                reply: None,
            });
        Ok(Subscriber::from_stream(stream))
    }

    fn publish_chat_interrupt(&self, _session_id: &str) {
        // Mono-instance: an interrupt is delivered in-process, never over the
        // transport, so there is no remote listener to signal.
    }

    async fn subscribe_chat_interrupt(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    fn publish_cancel_tools(&self, _session_id: &str) {
        // Mono-instance: no remote owner to signal.
    }

    async fn subscribe_cancel_tools(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn request_streaming_snapshot(&self, _session_id: &str) -> Option<StreamingSnapshot> {
        None
    }

    async fn subscribe_snapshot_requests(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }
    async fn publish(&self, _subject: &Subject, _payload: Vec<u8>) -> anyhow::Result<()> {
        // Mono-instance: no remote reply inbox to deliver to.
        Ok(())
    }

    async fn flush(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn transport_state(&self) -> TransportState {
        TransportState::Unknown
    }

    fn cancel_tools_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_tools", session_id)
    }

    fn cancel_task_subject(&self, session_id: &str) -> String {
        format!("chat.{}.cancel_task", session_id)
    }

    async fn subscribe_cancel_task(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }

    async fn subscribe_rpc_send(&self, _session_id: &str) -> anyhow::Result<Subscriber> {
        Ok(Subscriber::from_stream(
            futures::stream::pending::<Message>(),
        ))
    }
    async fn request_send_message(
        &self,
        _session_id: &str,
        _message: &str,
        _message_type: &str,
    ) -> Option<ChatRpcResponse> {
        None
    }

    fn request_cancel<'a>(
        &'a self,
        _subject: String,
        _payload: serde_json::Value,
        _is_provisional: &'a (dyn Fn(&[u8]) -> bool + Sync),
        _grace: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, RelayFailure>> + Send + 'a>,
    > {
        Box::pin(async move { Err(RelayFailure::NoResponders) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn in_memory_publish_reaches_a_subscriber_in_order() {
        let transport = InMemoryChatTransport::new();
        let mut sub = transport.subscribe_chat_events("s1").await.unwrap();

        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "one".into(),
            },
        );
        transport.publish_chat_event(
            "s1",
            ChatEvent::UserMessage {
                content: "two".into(),
            },
        );

        let first: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        let second: ChatEvent = serde_json::from_slice(&sub.next().await.unwrap().payload).unwrap();
        assert!(matches!(first, ChatEvent::UserMessage { content } if content == "one"));
        assert!(matches!(second, ChatEvent::UserMessage { content } if content == "two"));
    }

    #[tokio::test]
    async fn in_memory_cancel_reports_no_responders() {
        let transport = InMemoryChatTransport::new();
        let outcome = transport
            .request_cancel(
                "ignored".into(),
                serde_json::json!({}),
                &|_| false,
                Duration::from_secs(1),
            )
            .await;
        assert_eq!(outcome, Err(RelayFailure::NoResponders));
    }
}
