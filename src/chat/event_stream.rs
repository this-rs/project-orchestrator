//! Shared chat event-stream core: frame building and catch-up.
//!
//! The WebSocket handler and the REST/SSE event surface both need the same
//! "replay what was persisted, subscribe to the live broadcast BEFORE taking
//! the streaming snapshot (so nothing is lost), then forward live events with
//! fingerprint dedup" sequence. This module holds that core so no transport
//! duplicates it — the common execution contract (RFC cdd8c6ee).
//!
//! Persisted events carry a stable `seq` (and `id`); transient events
//! (`StreamDelta`, `StreamingStatus`) are never persisted and carry `seq: 0`
//! on the wire — a mid-stream join reconstructs them from the `partial_text`
//! snapshot, never from the durable cursor.
#![allow(unused_imports)]

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::broadcast;

use crate::chat::manager::ChatManager;
use crate::chat::types::ChatEvent;
use crate::events::{NatsEmitter, Subscriber};
use crate::neo4j::models::ChatEventRecord;

/// A server time on the chat wire: seconds since the epoch, with the
/// milliseconds as the fraction (`1760099999.123`). The REST history and every
/// chat event frame carry `created_at` in this unit.
pub(crate) fn wire_seconds(at: DateTime<Utc>) -> f64 {
    at.timestamp_millis() as f64 / 1000.0
}

/// A persisted event as a replay frame: its payload, flat, plus `seq`,
/// `replaying` and `created_at` (the time it was stored). `None` for a record
/// that must never reach the client: the `tool_result_spill` record carries the
/// full text of a spilled tool result, which stays server-side only.
pub(crate) fn replayed_frame(record: &ChatEventRecord) -> Option<Value> {
    if record.event_type == crate::chat::spill::SPILL_EVENT_TYPE {
        return None;
    }
    let created_at = serde_json::json!(wire_seconds(record.created_at));
    let frame = match serde_json::from_str::<Value>(&record.data) {
        Ok(Value::Object(mut obj)) => {
            obj.insert("seq".to_string(), serde_json::json!(record.seq));
            obj.insert("replaying".to_string(), serde_json::json!(true));
            obj.insert("created_at".to_string(), created_at);
            Value::Object(obj)
        }
        // Malformed record: wrap the raw payload in a data field.
        _ => serde_json::json!({
            "seq": record.seq,
            "type": record.event_type,
            "data": Value::String(record.data.clone()),
            "replaying": true,
            "created_at": created_at,
        }),
    };
    Some(frame)
}

/// A live event (local broadcast or relayed over NATS) as a frame: its payload
/// plus `seq: 0` and `created_at`, the time this server forwards it.
pub(crate) fn live_frame(event: &ChatEvent, now: DateTime<Utc>) -> serde_json::Result<Value> {
    let mut val = serde_json::to_value(event)?;
    if let Some(obj) = val.as_object_mut() {
        obj.insert("seq".to_string(), serde_json::json!(0));
        obj.insert(
            "created_at".to_string(),
            serde_json::json!(wire_seconds(now)),
        );
    }
    Ok(val)
}

/// A sink a transport implements to deliver chat frames: the WebSocket sends
/// text frames, the (future) SSE endpoint writes `data:` events. One trait, so
/// the catch-up core is written once.
#[async_trait::async_trait]
pub(crate) trait FrameSink {
    /// Deliver one frame. Returns `false` once the client is gone (the caller
    /// must then stop).
    async fn send_frame(&mut self, frame: Value) -> bool;
}

/// The live-subscription state a transport continues with after catch-up.
pub(crate) struct CatchUp {
    /// Local broadcast subscription, `None` when the session is not active
    /// locally (a remote instance owns it, or it is dormant).
    pub event_rx: Option<broadcast::Receiver<ChatEvent>>,
    /// Cross-instance subscription, used only when there is no local broadcast.
    pub nats_chat_sub: Option<Subscriber>,
    /// Fingerprints of the streaming snapshot, so live events from the
    /// subscribe→snapshot micro-window are not delivered twice.
    pub snapshot_fingerprints: HashSet<String>,
}

/// Catch-up failed because the client disconnected mid-replay.
#[derive(Debug)]
pub(crate) struct Disconnected;

/// Phase 1 (replay persisted events since `last_event`) + Phase 1.5a (subscribe
/// to the broadcast BEFORE the snapshot) + Phase 1.5b (streaming snapshot) +
/// pending-queue snapshot + `replay_complete`.
///
/// `sink` delivers every frame; the first `false` aborts catch-up with
/// [`Disconnected`]. The returned [`CatchUp`] is the live state the caller's
/// event loop continues with.
pub(crate) async fn catch_up<S: FrameSink>(
    chat_manager: &ChatManager,
    nats: Option<&NatsEmitter>,
    session_id: &str,
    last_event: i64,
    sink: &mut S,
) -> Result<CatchUp, Disconnected> {
    // ====================================================================
    // Phase 1: Replay persisted events since last_event.
    // ====================================================================
    // Skip the Neo4j query when last_event is very large (the client loads
    // history via REST and only needs the live streaming snapshot).
    const SKIP_REPLAY_THRESHOLD: i64 = 1_000_000_000_000;

    if last_event < SKIP_REPLAY_THRESHOLD {
        let events = chat_manager
            .get_events_since(session_id, last_event)
            .await
            .unwrap_or_default();

        if !events.is_empty() {
            for event in events {
                if let Some(msg) = replayed_frame(&event) {
                    if !sink.send_frame(msg).await {
                        return Err(Disconnected);
                    }
                }
            }
        } else if last_event == 0 {
            // Fallback: load from Nexus SDK for pre-migration sessions.
            match chat_manager
                .get_session_messages(session_id, None, None)
                .await
            {
                Ok(loaded) => {
                    for event in &loaded.events {
                        if let Some(msg) = replayed_frame(event) {
                            if !sink.send_frame(msg).await {
                                return Err(Disconnected);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        session_id = %session_id,
                        error = %e,
                        "Failed to load message history for replay"
                    );
                }
            }
        }
    }

    // ====================================================================
    // Phase 1.5a: subscribe BEFORE the snapshot so nothing is lost.
    // ====================================================================
    let event_rx = chat_manager.subscribe(session_id).await.ok();

    // NATS only when there is no local broadcast (else every event arrives twice).
    let nats_chat_sub: Option<Subscriber> = if event_rx.is_some() {
        None
    } else if let Some(nats) = nats {
        match nats.subscribe_chat_events(session_id).await {
            Ok(sub) => Some(sub),
            Err(e) => {
                tracing::warn!(session_id = %session_id, error = %e, "Failed to subscribe to NATS chat events");
                None
            }
        }
    } else {
        None
    };

    // ====================================================================
    // Phase 1.5b: streaming snapshot, taken AFTER subscribe.
    // ====================================================================
    let (mut is_currently_streaming, mut partial_text, mut streaming_events) =
        chat_manager.get_streaming_snapshot(session_id).await;

    if !is_currently_streaming {
        if let Some(nats) = nats {
            if let Some(snapshot) = nats.request_streaming_snapshot(session_id).await {
                is_currently_streaming = snapshot.is_streaming;
                partial_text = snapshot.partial_text;
                streaming_events = snapshot.events;
            }
        }
    }

    if is_currently_streaming {
        // Accumulated structured events (tool_use, tool_result, assistant_text…)
        // that occurred during the current stream.
        for event in &streaming_events {
            match serde_json::to_value(event) {
                Ok(mut val) => {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("seq".to_string(), serde_json::json!(0));
                        obj.insert("replaying".to_string(), serde_json::json!(true));
                    }
                    if !sink.send_frame(val).await {
                        return Err(Disconnected);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize streaming event: {}", e);
                }
            }
        }

        // Accumulated stream_delta text as a single partial_text snapshot.
        if !partial_text.is_empty() {
            let partial_msg = serde_json::json!({
                "type": "partial_text",
                "content": partial_text,
                "seq": 0,
                "replaying": true,
            });
            if !sink.send_frame(partial_msg).await {
                return Err(Disconnected);
            }
        }

        let status_msg = serde_json::json!({
            "type": "streaming_status",
            "is_streaming": true,
        });
        if !sink.send_frame(status_msg).await {
            return Err(Disconnected);
        }
    }

    // The messages held for this session, so a client that opens (or comes
    // back to) the conversation sees what is waiting. Sent even when empty.
    let held = match chat_manager.pending_queue_snapshot(session_id).await {
        Some(messages) => Some(messages),
        None => {
            let reached = chat_manager
                .pending_queue_op(session_id, &crate::chat::pending_queue::QueueOp::Snapshot)
                .await
                .unwrap_or(false);
            // Nobody holds the session: nothing can be waiting.
            (!reached).then(Vec::new)
        }
    };
    if let Some(messages) = held {
        let queue_msg = serde_json::json!({ "type": "pending_queue", "messages": messages });
        if !sink.send_frame(queue_msg).await {
            return Err(Disconnected);
        }
    }

    // Fingerprint dedup for the subscribe→snapshot micro-window.
    let snapshot_fingerprints: HashSet<String> = if is_currently_streaming {
        streaming_events
            .iter()
            .filter_map(|e| e.fingerprint())
            .collect()
    } else {
        HashSet::new()
    };

    let replay_complete = serde_json::json!({ "type": "replay_complete" });
    if !sink.send_frame(replay_complete).await {
        return Err(Disconnected);
    }

    Ok(CatchUp {
        event_rx,
        nats_chat_sub,
        snapshot_fingerprints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::manager::test_support;
    use crate::chat::manager::DeliveryRoute;
    use crate::test_helpers::mock_app_state;
    use uuid::Uuid;

    struct VecSink(Vec<Value>);
    #[async_trait::async_trait]
    impl FrameSink for VecSink {
        async fn send_frame(&mut self, frame: Value) -> bool {
            self.0.push(frame);
            true
        }
    }

    struct DisconnectSink;
    #[async_trait::async_trait]
    impl FrameSink for DisconnectSink {
        async fn send_frame(&mut self, _frame: Value) -> bool {
            false
        }
    }

    #[test]
    fn replayed_frame_wraps_a_malformed_payload_instead_of_dropping_it() {
        let record = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            seq: 7,
            event_type: "user_message".to_string(),
            data: "not-json".to_string(),
            created_at: chrono::Utc::now(),
        };
        let frame = replayed_frame(&record).expect("a malformed record still yields a frame");
        assert_eq!(frame["data"], "not-json");
        assert_eq!(frame["seq"], 7);
        assert_eq!(frame["replaying"], true);
    }

    #[tokio::test]
    async fn catch_up_subscribes_locally_and_sends_replay_complete() {
        let state = mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = Uuid::new_v4().to_string();
        test_support::insert_live_session_without_cli(&manager, &sid).await;

        let mut sink = VecSink(Vec::new());
        let caught = catch_up(&manager, None, &sid, 0, &mut sink)
            .await
            .expect("catch_up succeeds");
        assert!(
            caught.event_rx.is_some(),
            "a local session subscribes to the broadcast"
        );
        assert!(
            sink.0
                .iter()
                .any(|f| f.get("type").and_then(|t| t.as_str()) == Some("replay_complete")),
            "replay_complete is always sent"
        );
    }

    #[tokio::test]
    async fn catch_up_aborts_when_the_sink_gives_up() {
        let state = mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = Uuid::new_v4().to_string();
        test_support::insert_live_session_without_cli(&manager, &sid).await;

        let mut sink = DisconnectSink;
        let result = catch_up(&manager, None, &sid, 0, &mut sink).await;
        assert!(result.is_err(), "a sink that returns false aborts catch-up");
    }

    /// The id returned by `route_user_message` is the id persisted on the
    /// `user_message` event (the correlation the awaitable send relies on).
    #[tokio::test]
    async fn route_user_message_returns_the_persisted_turn_id() {
        let state = mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = Uuid::new_v4().to_string();
        test_support::insert_mock_cli_session(&manager, &sid).await;

        let (route, turn_id) = manager
            .route_user_message(&sid, "hello", None)
            .await
            .expect("route succeeds");
        assert!(matches!(route, DeliveryRoute::Local));

        let events = manager
            .get_events_since(&sid, 0)
            .await
            .expect("events are readable");
        let user_msg = events
            .iter()
            .find(|e| e.event_type == "user_message")
            .expect("the user_message is persisted");
        assert_eq!(
            user_msg.id, turn_id,
            "turn_id must equal the persisted user_message id"
        );
    }

    /// A streaming session rehydrates its accumulated text as a partial_text
    /// snapshot and announces streaming_status (the mid-turn join path).
    #[tokio::test]
    async fn catch_up_sends_the_streaming_snapshot() {
        let state = mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = Uuid::new_v4().to_string();
        test_support::insert_mock_cli_session(&manager, &sid).await;

        {
            let sessions = manager.active_sessions.read().await;
            let session = sessions.get(&sid).expect("session live");
            session
                .is_streaming
                .store(true, std::sync::atomic::Ordering::SeqCst);
            *session.streaming_text.lock().await = "partial".to_string();
        }

        let mut sink = VecSink(Vec::new());
        let _caught = catch_up(&manager, None, &sid, 0, &mut sink)
            .await
            .expect("catch_up succeeds");
        assert!(
            sink.0
                .iter()
                .any(|f| f.get("type").and_then(|t| t.as_str()) == Some("partial_text")),
            "the accumulated text is sent as partial_text"
        );
        assert!(
            sink.0
                .iter()
                .any(|f| f.get("type").and_then(|t| t.as_str()) == Some("streaming_status")),
            "streaming_status is sent"
        );
    }

    /// A message queued while streaming reuses the PendingMessage.id as its
    /// turn id: the returned id is the one later persisted when drained.
    #[tokio::test]
    async fn route_queued_user_message_reuses_pending_message_id() {
        let state = mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = Uuid::new_v4().to_string();
        test_support::insert_mock_cli_session(&manager, &sid).await;

        {
            let sessions = manager.active_sessions.read().await;
            let session = sessions.get(&sid).expect("session live");
            session
                .is_streaming
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        let (route, turn_id) = manager
            .route_queued_user_message(&sid, "later", None)
            .await
            .expect("queue succeeds");
        assert!(matches!(route, DeliveryRoute::Local));

        let sessions = manager.active_sessions.read().await;
        let session = sessions.get(&sid).expect("session live");
        let queue = session.pending_messages.lock().await;
        let held = queue
            .iter()
            .find(|m| matches!(m.kind, crate::chat::types::PendingMessageKind::User))
            .expect("the message is held while streaming");
        assert_eq!(
            held.id, turn_id,
            "PendingMessage.id must equal the returned turn_id"
        );
    }
}
