//! Drain loop — processes queued pending messages after stream ends.
//!
//! When `stream_response` completes, any messages queued in `pending_messages`
//! (from auto-continue, objective tracker, or concurrent user sends) are
//! drained one at a time: each message is persisted, broadcast, then
//! `stream_response` is called recursively to process it.

use super::manager::{ActiveSession, ChatManager};
use super::types::{ChatEvent, PendingMessage, PendingMessageKind};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
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
        let kind = next_msg.kind.clone();
        let msg_content = next_msg.content;

        info!(
            "Processing queued {} for session {} (queue was non-empty after stream)",
            match kind {
                PendingMessageKind::SystemHint => "system_hint",
                PendingMessageKind::User => "user_message",
                PendingMessageKind::BackgroundOutput => "background_output",
            },
            session_id
        );

        // Persist the event and update message count.
        // - SystemHint: persist as system_hint, no message_count bump
        // - User: persist as user_message, bump message_count
        // - BackgroundOutput: SKIP persist + SKIP broadcast — the OOB
        //   listener has already done both before pushing to the queue
        //   (see chat::oob_listener). Re-persisting would duplicate.
        let needs_persist_and_broadcast = !matches!(kind, PendingMessageKind::BackgroundOutput);
        if needs_persist_and_broadcast {
            if let Some(uuid) = session_uuid {
                if matches!(kind, PendingMessageKind::User) {
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

                let (event_type, event_data) = match kind {
                    PendingMessageKind::SystemHint => (
                        "system_hint".to_string(),
                        serde_json::to_string(&ChatEvent::SystemHint {
                            content: msg_content.clone(),
                        })
                        .unwrap_or_default(),
                    ),
                    PendingMessageKind::User => (
                        "user_message".to_string(),
                        serde_json::to_string(&ChatEvent::UserMessage {
                            content: msg_content.clone(),
                        })
                        .unwrap_or_default(),
                    ),
                    PendingMessageKind::BackgroundOutput => unreachable!(
                        "BackgroundOutput is filtered out by needs_persist_and_broadcast"
                    ),
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

            let chat_event = match kind {
                PendingMessageKind::SystemHint => ChatEvent::SystemHint {
                    content: msg_content.clone(),
                },
                PendingMessageKind::User => ChatEvent::UserMessage {
                    content: msg_content.clone(),
                },
                PendingMessageKind::BackgroundOutput => unreachable!(),
            };
            let _ = events_tx.send(chat_event.clone());
            if let Some(ref nats) = nats {
                nats.publish_chat_event(&session_id, chat_event);
            }
        } else {
            debug!(
                session_id = %session_id,
                "Drain: BackgroundOutput already persisted/broadcasted by OOB listener — \
                 forwarding content to stream_response as prompt"
            );
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
