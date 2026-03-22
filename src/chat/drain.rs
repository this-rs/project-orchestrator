//! Drain loop — processes queued pending messages after stream ends.
//!
//! When `stream_response` completes, any messages queued in `pending_messages`
//! (from auto-continue, objective tracker, or concurrent user sends) are
//! drained one at a time: each message is persisted, broadcast, then
//! `stream_response` is called recursively to process it.

use super::manager::{ActiveSession, ChatManager};
use super::types::{ChatEvent, PendingMessage, PendingMessageKind};
use crate::neo4j::models::ChatEventRecord;
use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager},
    InteractiveClient,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

/// Drain the pending_messages queue: pop the next message, persist it,
/// broadcast it, and recursively call `stream_response` to process it.
///
/// If the queue is empty but `has_pending` was true (race), clean up streaming status.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drain_pending_messages(
    has_pending: bool,
    client: Arc<Mutex<InteractiveClient>>,
    events_tx: broadcast::Sender<ChatEvent>,
    session_id: String,
    session_uuid: Option<Uuid>,
    graph: Arc<dyn GraphStore>,
    active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    interrupt_flag: Arc<AtomicBool>,
    memory_manager: Option<Arc<Mutex<ConversationMemoryManager>>>,
    context_injector: Option<Arc<ContextInjector>>,
    next_seq: Arc<AtomicI64>,
    pending_messages: Arc<Mutex<VecDeque<PendingMessage>>>,
    is_streaming: Arc<AtomicBool>,
    streaming_text: Arc<Mutex<String>>,
    streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
    event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    nats: Option<Arc<crate::events::NatsEmitter>>,
    shared_sdk_control_rx: Arc<
        tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<serde_json::Value>>>,
    >,
    auto_continue: Arc<AtomicBool>,
    retry_config: super::config::RetryConfig,
    enrichment_pipeline: Arc<super::enrichment::EnrichmentPipeline>,
    search: Arc<dyn SearchStore>,
) {
    // Pop the next message from the queue
    let next_message = {
        let mut queue = pending_messages.lock().await;
        queue.pop_front()
    };

    if let Some(next_msg) = next_message {
        let is_system_hint = next_msg.kind == PendingMessageKind::SystemHint;
        let msg_content = next_msg.content;

        info!(
            "Processing queued {} for session {} (queue was non-empty after stream)",
            if is_system_hint {
                "system_hint"
            } else {
                "user_message"
            },
            session_id
        );

        // Persist the event and update message count (only for user messages)
        if let Some(uuid) = session_uuid {
            if !is_system_hint {
                if let Ok(Some(node)) = graph.get_chat_session(uuid).await {
                    let _ = graph
                        .update_chat_session(
                            uuid,
                            None,
                            None,
                            Some(node.message_count + 1),
                            None,
                            None,
                            None,
                        )
                        .await;
                }
            }

            let (event_type, event_data) = if is_system_hint {
                (
                    "system_hint".to_string(),
                    serde_json::to_string(&ChatEvent::SystemHint {
                        content: msg_content.clone(),
                    })
                    .unwrap_or_default(),
                )
            } else {
                (
                    "user_message".to_string(),
                    serde_json::to_string(&ChatEvent::UserMessage {
                        content: msg_content.clone(),
                    })
                    .unwrap_or_default(),
                )
            };

            let event_record = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: uuid,
                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                event_type,
                data: event_data,
                created_at: chrono::Utc::now(),
            };
            let _ = graph.store_chat_events(uuid, vec![event_record]).await;
        }

        // Emit the appropriate event on broadcast + NATS
        let chat_event = if is_system_hint {
            ChatEvent::SystemHint {
                content: msg_content.clone(),
            }
        } else {
            ChatEvent::UserMessage {
                content: msg_content.clone(),
            }
        };
        let _ = events_tx.send(chat_event.clone());
        if let Some(ref nats) = nats {
            nats.publish_chat_event(&session_id, chat_event);
        }

        // Recursive call to process the queued message
        Box::pin(ChatManager::stream_response(
            client,
            events_tx,
            msg_content,
            session_id,
            graph,
            active_sessions,
            interrupt_flag,
            memory_manager,
            context_injector,
            next_seq,
            pending_messages,
            is_streaming,
            streaming_text,
            streaming_events,
            event_emitter,
            nats,
            shared_sdk_control_rx,
            auto_continue,
            retry_config,
            enrichment_pipeline,
            search,
        ))
        .await;
    } else if has_pending {
        // Race: has_pending was true but pop_front returned None.
        // Set is_streaming=false now since there's nothing to process.
        is_streaming.store(false, Ordering::SeqCst);
        let _ = events_tx.send(ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        if let Some(ref nats) = nats {
            nats.publish_chat_event(
                &session_id,
                ChatEvent::StreamingStatus {
                    is_streaming: false,
                },
            );
        }
        if let Some(ref emitter) = event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &session_id,
                serde_json::json!({ "is_streaming": false }),
                None,
            );
        }
        streaming_text.lock().await.clear();
        streaming_events.lock().await.clear();
        debug!(
            "Stream completed for session {} (pending race resolved)",
            session_id
        );
    }
}
