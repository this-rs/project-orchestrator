//! ChatManager — orchestrates Claude Code CLI sessions via Nexus SDK
//!
//! Manages active InteractiveClient sessions with auto-resume for inactive ones.
//!
//! Architecture:
//! - Each session spawns an `InteractiveClient` (Nexus SDK) subprocess
//! - Messages are streamed via `broadcast::channel` to WebSocket subscribers
//! - Structured events are persisted in Neo4j with sequence numbers for replay
//! - Inactive sessions are persisted in Neo4j with `cli_session_id` for resume
//! - A cleanup task periodically closes timed-out sessions

use super::config::ChatConfig;
use super::types::{
    truncate_snippet, ChatEvent, ChatEventPage, ChatRequest, CreateSessionResponse,
    MessageSearchHit, MessageSearchResult,
};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::models::ChatSessionNode;
use crate::neo4j::GraphStore;
use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager, MemoryConfig},
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    PermissionMode, StreamDelta, StreamEventData,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::expand_tilde;

/// Broadcast channel buffer size for WebSocket subscribers
const BROADCAST_BUFFER: usize = 256;

/// An active chat session with a live Claude CLI subprocess
pub struct ActiveSession {
    /// Persistent broadcast sender — one per session lifetime, NOT replaced per message
    pub events_tx: broadcast::Sender<ChatEvent>,
    /// When the session was last active
    pub last_activity: Instant,
    /// The CLI session ID (for persistence / resume)
    pub cli_session_id: Option<String>,
    /// Handle to the InteractiveClient (behind Mutex for &mut access)
    pub client: Arc<Mutex<InteractiveClient>>,
    /// Flag to signal the stream loop to stop and release the client lock
    pub interrupt_flag: Arc<AtomicBool>,
    /// Nexus conversation memory manager (records messages for persistence)
    pub memory_manager: Option<Arc<Mutex<ConversationMemoryManager>>>,
    /// Monotonically increasing sequence number for persisted events
    pub next_seq: Arc<AtomicI64>,
    /// Queue of messages waiting to be sent (received while streaming)
    pub pending_messages: Arc<Mutex<VecDeque<String>>>,
    /// Whether a stream is currently in progress
    pub is_streaming: Arc<AtomicBool>,
    /// Accumulated text from stream_delta during the current stream (for mid-stream join)
    pub streaming_text: Arc<Mutex<String>>,
    /// Accumulated structured events during the current stream (for mid-stream join).
    /// Contains all non-StreamDelta events (ToolUse, ToolResult, AssistantText, etc.)
    /// that haven't been persisted yet. Cleared at stream start/end.
    pub streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
}

/// Manages chat sessions and their lifecycle
pub struct ChatManager {
    pub(crate) graph: Arc<dyn GraphStore>,
    #[allow(dead_code)]
    pub(crate) search: Arc<dyn SearchStore>,
    pub(crate) config: ChatConfig,
    pub(crate) active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    /// Nexus memory injector for conversation persistence
    pub(crate) context_injector: Option<Arc<ContextInjector>>,
    /// Memory config (for creating ConversationMemoryManagers)
    pub(crate) memory_config: Option<MemoryConfig>,
    /// Event emitter for CRUD events (streaming status changes)
    pub(crate) event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    /// Optional NATS emitter for cross-instance chat event publishing
    pub(crate) nats: Option<Arc<crate::events::NatsEmitter>>,
}

impl ChatManager {
    /// Create a ChatManager without memory support (for tests or when Meilisearch is unavailable)
    pub fn new_without_memory(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            context_injector: None,
            memory_config: None,
            event_emitter: None,
            nats: None,
        }
    }

    /// Create a new ChatManager with conversation memory support
    pub async fn new(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        // Initialize ContextInjector for conversation memory persistence
        let memory_config = MemoryConfig {
            meilisearch_url: config.meilisearch_url.clone(),
            meilisearch_key: Some(config.meilisearch_key.clone()),
            enabled: true,
            ..MemoryConfig::default()
        };
        let context_injector = match ContextInjector::new(memory_config.clone()).await {
            Ok(injector) => {
                info!("ContextInjector initialized for conversation memory");
                Some(Arc::new(injector))
            }
            Err(e) => {
                warn!("Failed to initialize ContextInjector: {} — message history will be unavailable", e);
                None
            }
        };

        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            context_injector,
            memory_config: Some(memory_config),
            event_emitter: None,
            nats: None,
        }
    }

    /// Set the event emitter for CRUD events (streaming status notifications)
    pub fn with_event_emitter(mut self, emitter: Arc<dyn crate::events::EventEmitter>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Set the NATS emitter for cross-instance chat event publishing.
    ///
    /// When configured, ChatEvents are published to NATS alongside the local
    /// broadcast channel, enabling multi-instance real-time sync.
    /// Interrupts are also propagated via NATS to all instances.
    pub fn with_nats(mut self, nats: Arc<crate::events::NatsEmitter>) -> Self {
        self.nats = Some(nats);
        self
    }

    /// Spawn a background task that listens for NATS interrupt signals for a session.
    ///
    /// When another instance publishes an interrupt for this session via NATS,
    /// the listener sets the local `interrupt_flag` so the stream loop breaks.
    /// No-op if NATS is not configured.
    fn spawn_nats_interrupt_listener(
        &self,
        session_id: &str,
        interrupt_flag: Arc<AtomicBool>,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_interrupt(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS interrupt for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            while let Some(_msg) = subscriber.next().await {
                // Check if session is still active — stop listener if session was removed
                let session_exists = {
                    let sessions = active_sessions.read().await;
                    sessions.contains_key(&session_id)
                };
                if !session_exists {
                    debug!(
                        "Session {} no longer active, stopping NATS interrupt listener",
                        session_id
                    );
                    break;
                }

                // Guard: don't re-interrupt if the flag is already set
                if interrupt_flag.load(Ordering::SeqCst) {
                    debug!(
                        "Interrupt flag already set for session {}, ignoring NATS interrupt",
                        session_id
                    );
                    continue;
                }

                info!(
                    "NATS interrupt received for session {}, setting interrupt flag",
                    session_id
                );
                interrupt_flag.store(true, Ordering::SeqCst);
            }

            debug!("NATS interrupt listener stopped for session {}", session_id);
        });
    }

    /// Spawn a background task that responds to NATS snapshot requests for a session.
    ///
    /// When a remote instance needs to do a mid-stream join, it sends a NATS request
    /// on `events.chat.{session_id}.snapshot`. This listener replies with the current
    /// streaming snapshot (partial text + structured events).
    /// Stops when the session is removed from active_sessions.
    fn spawn_nats_snapshot_responder(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_snapshot_requests(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS snapshot requests for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            while let Some(msg) = subscriber.next().await {
                // Build snapshot from active session state
                let snapshot = {
                    let sessions = active_sessions.read().await;
                    match sessions.get(&session_id) {
                        Some(session) => {
                            let is_streaming = session.is_streaming.load(Ordering::SeqCst);
                            let text = session.streaming_text.lock().await.clone();
                            let events = session.streaming_events.lock().await.clone();
                            Some(crate::events::StreamingSnapshot {
                                is_streaming,
                                partial_text: text,
                                events,
                            })
                        }
                        None => {
                            // Session no longer active — stop responder
                            debug!(
                                "Session {} no longer active, stopping snapshot responder",
                                session_id
                            );
                            break;
                        }
                    }
                };

                if let Some(snapshot) = snapshot {
                    // Reply to the requester
                    if let Some(reply_to) = msg.reply {
                        match serde_json::to_vec(&snapshot) {
                            Ok(payload) => {
                                if let Err(e) =
                                    nats.client().publish(reply_to, payload.into()).await
                                {
                                    warn!(
                                        "Failed to reply with snapshot for session {}: {}",
                                        session_id, e
                                    );
                                } else {
                                    debug!(
                                        session_id = %session_id,
                                        is_streaming = snapshot.is_streaming,
                                        text_len = snapshot.partial_text.len(),
                                        events_count = snapshot.events.len(),
                                        "Replied with streaming snapshot"
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to serialize snapshot for session {}: {}",
                                    session_id, e
                                );
                            }
                        }
                    }
                }
            }

            debug!("NATS snapshot responder stopped for session {}", session_id);
        });
    }

    /// Spawn a background task that listens for NATS RPC send requests for a session.
    ///
    /// When another instance wants to send a message to a session owned by this instance,
    /// it publishes a `ChatRpcRequest` to `rpc.chat.{session_id}.send`.
    /// This listener processes the request locally (queue if streaming, or persist+stream)
    /// and replies with `ChatRpcResponse`.
    ///
    /// No-op if NATS is not configured.
    fn spawn_nats_rpc_listener(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();
        let graph = self.graph.clone();
        let context_injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_rpc_send(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS RPC send for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            info!(
                session_id = %session_id,
                "NATS RPC send listener started"
            );

            while let Some(msg) = subscriber.next().await {
                // Parse the RPC request
                let request: crate::events::ChatRpcRequest =
                    match serde_json::from_slice(&msg.payload) {
                        Ok(req) => req,
                        Err(e) => {
                            warn!(
                                "Failed to parse NATS RPC request for session {}: {}",
                                session_id, e
                            );
                            // Reply with error if possible
                            if let Some(reply_to) = msg.reply {
                                let resp = crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(format!("Invalid request: {}", e)),
                                };
                                if let Ok(payload) = serde_json::to_vec(&resp) {
                                    let _ = nats.client().publish(reply_to, payload.into()).await;
                                }
                            }
                            continue;
                        }
                    };

                debug!(
                    session_id = %session_id,
                    message_type = %request.message_type,
                    "NATS RPC send request received"
                );

                // Get session state — check if still active locally
                let session_state = {
                    let mut sessions = active_sessions.write().await;
                    match sessions.get_mut(&session_id) {
                        Some(session) => {
                            session.last_activity = Instant::now();
                            session.interrupt_flag.store(false, Ordering::SeqCst);
                            Some((
                                session.client.clone(),
                                session.events_tx.clone(),
                                session.interrupt_flag.clone(),
                                session.memory_manager.clone(),
                                session.next_seq.clone(),
                                session.pending_messages.clone(),
                                session.is_streaming.clone(),
                                session.streaming_text.clone(),
                                session.streaming_events.clone(),
                            ))
                        }
                        None => None,
                    }
                };

                let response = match session_state {
                    None => {
                        // Session no longer active — reply error and stop listener
                        debug!(
                            "Session {} no longer active, stopping NATS RPC listener",
                            session_id
                        );
                        let resp = crate::events::ChatRpcResponse {
                            success: false,
                            error: Some("Session not active on this instance".to_string()),
                        };
                        if let Some(reply_to) = msg.reply {
                            if let Ok(payload) = serde_json::to_vec(&resp) {
                                let _ = nats.client().publish(reply_to, payload.into()).await;
                            }
                        }
                        break;
                    }
                    Some((
                        client,
                        events_tx,
                        interrupt_flag,
                        memory_manager,
                        next_seq,
                        pending_messages,
                        is_streaming,
                        streaming_text,
                        streaming_events,
                    )) => {
                        let message = &request.message;

                        // If streaming → queue the message (will be drained by stream_response)
                        if is_streaming.load(Ordering::SeqCst) {
                            info!(
                                "Stream in progress for session {} (via NATS RPC), queuing message",
                                session_id
                            );
                            let mut queue = pending_messages.lock().await;
                            queue.push_back(message.clone());
                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        } else {
                            // Not streaming — persist user_message, broadcast, spawn stream
                            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                                // Update message count
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

                                // Persist user_message event
                                let user_event = crate::neo4j::models::ChatEventRecord {
                                    id: Uuid::new_v4(),
                                    session_id: uuid,
                                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                                    event_type: "user_message".to_string(),
                                    data: serde_json::to_string(
                                        &serde_json::json!({"content": message}),
                                    )
                                    .unwrap_or_default(),
                                    created_at: chrono::Utc::now(),
                                };
                                let _ = graph.store_chat_events(uuid, vec![user_event]).await;
                            }

                            // Broadcast user_message locally + NATS
                            let user_msg_event = ChatEvent::UserMessage {
                                content: message.clone(),
                            };
                            let _ = events_tx.send(user_msg_event.clone());
                            nats.publish_chat_event(&session_id, user_msg_event);

                            // Spawn stream_response
                            let session_id_clone = session_id.clone();
                            let graph_clone = graph.clone();
                            let active_sessions_clone = active_sessions.clone();
                            let prompt = message.clone();
                            let injector = context_injector.clone();
                            let event_emitter_clone = event_emitter.clone();
                            let nats_clone = Some(nats.clone());

                            tokio::spawn(async move {
                                Self::stream_response(
                                    client,
                                    events_tx,
                                    prompt,
                                    session_id_clone,
                                    graph_clone,
                                    active_sessions_clone,
                                    interrupt_flag,
                                    memory_manager,
                                    injector,
                                    next_seq,
                                    pending_messages,
                                    is_streaming,
                                    streaming_text,
                                    streaming_events,
                                    event_emitter_clone,
                                    nats_clone,
                                )
                                .await;
                            });

                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        }
                    }
                };

                // Reply to the requester
                if let Some(reply_to) = msg.reply {
                    match serde_json::to_vec(&response) {
                        Ok(payload) => {
                            if let Err(e) = nats.client().publish(reply_to, payload.into()).await {
                                warn!(
                                    "Failed to reply to NATS RPC for session {}: {}",
                                    session_id, e
                                );
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Failed to serialize RPC response for session {}: {}",
                                session_id, e
                            );
                        }
                    }
                }
            }

            debug!("NATS RPC send listener stopped for session {}", session_id);
        });
    }

    /// Resolve the model to use: request > config default
    pub fn resolve_model(&self, request_model: Option<&str>) -> String {
        request_model
            .map(|m| m.to_string())
            .unwrap_or_else(|| self.config.default_model.clone())
    }

    /// Build the system prompt with project context.
    ///
    /// Two-layer architecture:
    /// 1. Hardcoded base (BASE_SYSTEM_PROMPT) — protocols, data model, git, statuses
    /// 2. Dynamic context — oneshot Opus refines raw Neo4j data, fallback to markdown
    pub async fn build_system_prompt(
        &self,
        project_slug: Option<&str>,
        user_message: &str,
    ) -> String {
        use super::prompt::{
            assemble_prompt, context_to_json, context_to_markdown, fetch_project_context,
            BASE_SYSTEM_PROMPT,
        };

        // No project → base prompt only
        let Some(slug) = project_slug else {
            return BASE_SYSTEM_PROMPT.to_string();
        };

        // Fetch raw context from Neo4j
        let ctx = match fetch_project_context(&self.graph, slug).await {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!(
                    "Failed to fetch project context for '{}': {} — using base prompt only",
                    slug, e
                );
                return BASE_SYSTEM_PROMPT.to_string();
            }
        };

        // Try oneshot Opus refinement (with tool catalog for tool-aware prompting)
        let context_json = context_to_json(&ctx);
        let tools_catalog_json = super::prompt::tools_catalog_to_json(super::prompt::TOOL_GROUPS);
        let dynamic_section = match self
            .refine_context_with_oneshot(user_message, &context_json, &tools_catalog_json)
            .await
        {
            Ok(refined) => refined,
            Err(e) => {
                warn!(
                    "Oneshot context refinement failed: {} — using markdown fallback",
                    e
                );
                context_to_markdown(&ctx)
            }
        };

        assemble_prompt(BASE_SYSTEM_PROMPT, &dynamic_section)
    }

    /// Use a oneshot Opus call to refine raw project context into a concise,
    /// relevant contextual section for the system prompt.
    async fn refine_context_with_oneshot(
        &self,
        user_message: &str,
        context_json: &str,
        tools_catalog_json: &str,
    ) -> Result<String> {
        use super::prompt::build_refinement_prompt;

        let refinement_prompt =
            build_refinement_prompt(user_message, context_json, tools_catalog_json);

        // Build options: no MCP server, max_turns=1, just text generation
        #[allow(deprecated)]
        let options = ClaudeCodeOptions::builder()
            .model(&self.config.prompt_builder_model)
            .system_prompt("Tu es un assistant qui construit des sections de contexte concises.")
            .permission_mode(PermissionMode::BypassPermissions)
            .max_turns(1)
            .build();

        let mut client = InteractiveClient::new(options)
            .map_err(|e| anyhow!("Failed to create oneshot client: {}", e))?;

        client
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect oneshot client: {}", e))?;

        let messages = client
            .send_and_receive(refinement_prompt)
            .await
            .map_err(|e| anyhow!("Oneshot send_and_receive failed: {}", e))?;

        // Extract text from assistant messages
        let mut result = String::new();
        for msg in &messages {
            if let Message::Assistant { message, .. } = msg {
                for block in &message.content {
                    if let ContentBlock::Text(text) = block {
                        result.push_str(&text.text);
                    }
                }
            }
        }

        let _ = client.disconnect().await;

        if result.is_empty() {
            return Err(anyhow!("Oneshot returned empty response"));
        }

        Ok(result)
    }

    /// Check if a session is currently active (subprocess alive)
    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.active_sessions.read().await.contains_key(session_id)
    }

    // ========================================================================
    // ClaudeCodeOptions builder
    // ========================================================================

    /// Build `ClaudeCodeOptions` for a new or resumed session
    #[allow(deprecated)]
    pub fn build_options(
        &self,
        cwd: &str,
        model: &str,
        system_prompt: &str,
        resume_id: Option<&str>,
    ) -> ClaudeCodeOptions {
        // Expand tilde in cwd (shell doesn't expand ~ when passed via Command)
        let cwd = expand_tilde(cwd);
        let mcp_path = self.config.mcp_server_path.to_string_lossy().to_string();

        let mut env = HashMap::new();
        env.insert("NEO4J_URI".into(), self.config.neo4j_uri.clone());
        env.insert("NEO4J_USER".into(), self.config.neo4j_user.clone());
        env.insert("NEO4J_PASSWORD".into(), self.config.neo4j_password.clone());
        env.insert(
            "MEILISEARCH_URL".into(),
            self.config.meilisearch_url.clone(),
        );
        env.insert(
            "MEILISEARCH_KEY".into(),
            self.config.meilisearch_key.clone(),
        );

        let mcp_config = McpServerConfig::Stdio {
            command: mcp_path,
            args: None,
            env: Some(env),
        };

        let mut builder = ClaudeCodeOptions::builder()
            .model(model)
            .cwd(cwd)
            .system_prompt(system_prompt)
            .permission_mode(PermissionMode::BypassPermissions)
            .max_turns(self.config.max_turns)
            .include_partial_messages(true)
            .add_mcp_server("project-orchestrator", mcp_config);

        if let Some(id) = resume_id {
            builder = builder.resume(id);
        }

        builder.build()
    }

    // ========================================================================
    // Message → ChatEvent conversion
    // ========================================================================

    /// Convert a Nexus SDK `Message` to a list of `ChatEvent`s
    pub fn message_to_events(msg: &Message) -> Vec<ChatEvent> {
        match msg {
            Message::Assistant { message, .. } => {
                let mut events = Vec::new();
                for block in &message.content {
                    match block {
                        ContentBlock::Text(t) => {
                            events.push(ChatEvent::AssistantText {
                                content: t.text.clone(),
                            });
                        }
                        ContentBlock::Thinking(t) => {
                            events.push(ChatEvent::Thinking {
                                content: t.thinking.clone(),
                            });
                        }
                        ContentBlock::ToolUse(t) => {
                            events.push(ChatEvent::ToolUse {
                                id: t.id.clone(),
                                tool: t.name.clone(),
                                input: t.input.clone(),
                            });
                        }
                        ContentBlock::ToolResult(t) => {
                            let result = match &t.content {
                                Some(ContentValue::Text(s)) => serde_json::Value::String(s.clone()),
                                Some(ContentValue::Structured(v)) => {
                                    serde_json::Value::Array(v.clone())
                                }
                                None => serde_json::Value::Null,
                            };
                            events.push(ChatEvent::ToolResult {
                                id: t.tool_use_id.clone(),
                                result,
                                is_error: t.is_error.unwrap_or(false),
                            });
                        }
                    }
                }
                events
            }
            Message::Result {
                session_id,
                duration_ms,
                total_cost_usd,
                ..
            } => {
                vec![ChatEvent::Result {
                    session_id: session_id.clone(),
                    duration_ms: *duration_ms as u64,
                    cost_usd: *total_cost_usd,
                }]
            }
            Message::StreamEvent { event, .. } => match event {
                StreamEventData::ContentBlockDelta {
                    delta: StreamDelta::TextDelta { text },
                    ..
                } => {
                    vec![ChatEvent::StreamDelta { text: text.clone() }]
                }
                // Extract tool_use from ContentBlockStart — this is where tool calls
                // first appear in the stream (before the AssistantMessage is finalized).
                // The content_block is raw JSON: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                StreamEventData::ContentBlockStart { content_block, .. } => {
                    if content_block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                        let id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = content_block
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        vec![ChatEvent::ToolUse {
                            id,
                            tool: name,
                            input,
                        }]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            },
            Message::System { subtype, data } => {
                debug!("System message: {} — {:?}", subtype, data);
                vec![]
            }
            Message::User { message, .. } => {
                // User messages with content_blocks contain tool_result blocks
                // from the CLI's tool execution. Extract them as ChatEvent::ToolResult.
                if let Some(blocks) = &message.content_blocks {
                    let mut events = Vec::new();
                    for block in blocks {
                        if let ContentBlock::ToolResult(t) = block {
                            let result = match &t.content {
                                Some(ContentValue::Text(s)) => serde_json::Value::String(s.clone()),
                                Some(ContentValue::Structured(v)) => {
                                    serde_json::Value::Array(v.clone())
                                }
                                None => serde_json::Value::Null,
                            };
                            events.push(ChatEvent::ToolResult {
                                id: t.tool_use_id.clone(),
                                result,
                                is_error: t.is_error.unwrap_or(false),
                            });
                        }
                    }
                    events
                } else {
                    vec![]
                }
            }
        }
    }

    // ========================================================================
    // Session lifecycle
    // ========================================================================

    /// Create a new chat session: persist to Neo4j, spawn CLI subprocess, start streaming
    pub async fn create_session(&self, request: &ChatRequest) -> Result<CreateSessionResponse> {
        // Check max sessions
        {
            let sessions = self.active_sessions.read().await;
            if sessions.len() >= self.config.max_sessions {
                return Err(anyhow!(
                    "Maximum number of active sessions reached ({})",
                    self.config.max_sessions
                ));
            }
        }

        let session_id = Uuid::new_v4();
        let model = self.resolve_model(request.model.as_deref());
        let system_prompt = self
            .build_system_prompt(request.project_slug.as_deref(), &request.message)
            .await;

        // Persist session in Neo4j
        let session_node = ChatSessionNode {
            id: session_id,
            cli_session_id: None,
            project_slug: request.project_slug.clone(),
            cwd: request.cwd.clone(),
            title: None,
            model: model.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
        };
        self.graph
            .create_chat_session(&session_node)
            .await
            .context("Failed to persist chat session")?;

        // Build options and create InteractiveClient
        let options = self.build_options(&request.cwd, &model, &system_prompt, None);
        let mut client = InteractiveClient::new(options)
            .map_err(|e| anyhow!("Failed to create InteractiveClient: {}", e))?;

        client
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect InteractiveClient: {}", e))?;

        // Create ConversationMemoryManager for message recording
        let memory_manager = if let Some(ref mem_config) = self.memory_config {
            let mm = ConversationMemoryManager::new(mem_config.clone());
            let conversation_id = mm.conversation_id().to_string();
            debug!(
                "Created ConversationMemoryManager for session {} with conversation_id {}",
                session_id, conversation_id
            );

            // Persist conversation_id in Neo4j
            let _ = self
                .graph
                .update_chat_session(
                    session_id,
                    None,
                    None,
                    None,
                    None,
                    Some(conversation_id),
                    None,
                )
                .await;

            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        info!("Created chat session {} with model {}", session_id, model);

        // Create broadcast channel
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);
        let client = Arc::new(Mutex::new(client));

        // Initialize next_seq (new session = start at 1)
        let next_seq = Arc::new(AtomicI64::new(1));
        let pending_messages = Arc::new(Mutex::new(VecDeque::new()));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let streaming_text = Arc::new(Mutex::new(String::new()));
        let streaming_events = Arc::new(Mutex::new(Vec::new()));

        // Register active session
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            let interrupt_flag = Arc::new(AtomicBool::new(false));
            sessions.insert(
                session_id.to_string(),
                ActiveSession {
                    events_tx: events_tx.clone(),
                    last_activity: Instant::now(),
                    cli_session_id: None,
                    client: client.clone(),
                    interrupt_flag: interrupt_flag.clone(),
                    memory_manager: memory_manager.clone(),
                    next_seq: next_seq.clone(),
                    pending_messages: pending_messages.clone(),
                    is_streaming: is_streaming.clone(),
                    streaming_text: streaming_text.clone(),
                    streaming_events: streaming_events.clone(),
                },
            );
            interrupt_flag
        };

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            &session_id.to_string(),
            interrupt_flag.clone(),
            self.active_sessions.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(&session_id.to_string(), self.active_sessions.clone());

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(&session_id.to_string(), self.active_sessions.clone());

        // Persist the initial user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&serde_json::json!({"content": &request.message}))
                .unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let _ = self
            .graph
            .store_chat_events(session_id, vec![user_event])
            .await;

        // Emit user_message on local broadcast + NATS (so all clients see it)
        let user_msg_event = ChatEvent::UserMessage {
            content: request.message.clone(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(&session_id.to_string(), user_msg_event);
        }

        // Emit CRUD event so other instances (via NATS) know a session was created
        if let Some(ref emitter) = self.event_emitter {
            emitter.emit_created(
                crate::events::EntityType::ChatSession,
                &session_id.to_string(),
                serde_json::json!({
                    "project_slug": request.project_slug,
                    "cwd": request.cwd,
                    "model": model,
                }),
                None,
            );
        }

        // Auto-generate title and preview from the first user message
        {
            let msg = &request.message;
            let title = if msg.chars().count() > 80 {
                let truncated: String = msg.chars().take(77).collect();
                format!("{}...", truncated.trim_end())
            } else {
                msg.to_string()
            };
            let preview = if msg.chars().count() > 200 {
                let truncated: String = msg.chars().take(197).collect();
                format!("{}...", truncated.trim_end())
            } else {
                msg.to_string()
            };
            let _ = self
                .graph
                .update_chat_session(
                    session_id,
                    None,
                    Some(title),
                    None,
                    None,
                    None,
                    Some(preview),
                )
                .await;
        }

        // Send the initial message and start streaming in a background task
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let message = request.message.clone();
        let events_tx_clone = events_tx.clone();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx_clone,
                message,
                session_id_str.clone(),
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
            )
            .await;
        });

        Ok(CreateSessionResponse {
            session_id: session_id.to_string(),
            stream_url: format!("/ws/chat/{}", session_id),
        })
    }

    /// Internal: send a message to the client and stream the response to broadcast
    #[allow(clippy::too_many_arguments)]
    async fn stream_response(
        client: Arc<Mutex<InteractiveClient>>,
        events_tx: broadcast::Sender<ChatEvent>,
        prompt: String,
        session_id: String,
        graph: Arc<dyn GraphStore>,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        interrupt_flag: Arc<AtomicBool>,
        memory_manager: Option<Arc<Mutex<ConversationMemoryManager>>>,
        context_injector: Option<Arc<ContextInjector>>,
        next_seq: Arc<AtomicI64>,
        pending_messages: Arc<Mutex<VecDeque<String>>>,
        is_streaming: Arc<AtomicBool>,
        streaming_text: Arc<Mutex<String>>,
        streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
        event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
    ) {
        // Helper closure: emit a ChatEvent to local broadcast + NATS (if configured)
        let emit_chat = |event: ChatEvent,
                         tx: &broadcast::Sender<ChatEvent>,
                         nats: &Option<Arc<crate::events::NatsEmitter>>,
                         sid: &str| {
            let _ = tx.send(event.clone());
            if let Some(ref nats) = nats {
                nats.publish_chat_event(sid, event);
            }
        };

        is_streaming.store(true, Ordering::SeqCst);
        // Broadcast streaming_status to all connected clients (multi-tab support)
        emit_chat(
            ChatEvent::StreamingStatus { is_streaming: true },
            &events_tx,
            &nats,
            &session_id,
        );
        // Emit CRUD event for session list live refresh
        if let Some(ref emitter) = event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &session_id,
                serde_json::json!({ "is_streaming": true }),
                None,
            );
        }
        // Clear streaming buffers for the new stream
        streaming_text.lock().await.clear();
        streaming_events.lock().await.clear();

        // Record user message in memory manager
        if let Some(ref mm) = memory_manager {
            let mut mm = mm.lock().await;
            mm.record_user_message(&prompt);
        }

        // Events are persisted in Neo4j — the WebSocket replay handles late-joining clients.

        // Use send_and_receive_stream() for real-time token streaming.
        // The stream's lifetime is tied to the MutexGuard, so we hold the client lock
        // during the stream. However, the underlying transport uses separate stdin/stdout
        // channels, so other operations (like interrupt) can still proceed via the
        // interrupt_flag mechanism.
        let mut assistant_text_parts: Vec<String> = Vec::new();
        let mut events_to_persist: Vec<ChatEventRecord> = Vec::new();
        // Maps tool_use ID → index in events_to_persist, so we can update
        // the persisted record when AssistantMessage provides the full input.
        let mut emitted_tool_use_ids: std::collections::HashMap<String, Option<usize>> =
            std::collections::HashMap::new();
        let session_uuid = Uuid::parse_str(&session_id).ok();

        {
            let mut c = client.lock().await;
            let stream_result = c.send_and_receive_stream(prompt).await;

            let mut stream = match stream_result {
                Ok(s) => std::pin::pin!(s),
                Err(e) => {
                    error!("Error starting stream for session {}: {}", session_id, e);
                    emit_chat(
                        ChatEvent::Error {
                            message: format!("Error: {}", e),
                        },
                        &events_tx,
                        &nats,
                        &session_id,
                    );
                    is_streaming.store(false, Ordering::SeqCst);
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
                    return;
                }
            };

            while let Some(result) = stream.next().await {
                // Check interrupt flag at each iteration
                if interrupt_flag.load(Ordering::SeqCst) {
                    info!(
                        "Interrupt flag detected during stream for session {}",
                        session_id
                    );
                    break;
                }

                match result {
                    Ok(ref msg) => {
                        // Handle StreamEvent — emit StreamDelta for text tokens directly
                        // stream_delta are NOT persisted (too many writes)
                        if let Message::StreamEvent {
                            event:
                                StreamEventData::ContentBlockDelta {
                                    delta: StreamDelta::TextDelta { ref text },
                                    ..
                                },
                            ..
                        } = msg
                        {
                            // Accumulate for mid-stream join snapshot
                            streaming_text.lock().await.push_str(text);
                            emit_chat(
                                ChatEvent::StreamDelta { text: text.clone() },
                                &events_tx,
                                &nats,
                                &session_id,
                            );
                            continue;
                        }

                        // Extract cli_session_id from Result message
                        if let Message::Result {
                            session_id: ref cli_sid,
                            total_cost_usd: ref cost,
                            ..
                        } = msg
                        {
                            // Update Neo4j with cli_session_id and cost
                            if let Some(uuid) = session_uuid {
                                let _ = graph
                                    .update_chat_session(
                                        uuid,
                                        Some(cli_sid.clone()),
                                        None,
                                        Some(1),
                                        *cost,
                                        None,
                                        None,
                                    )
                                    .await;
                            }

                            // Update active session's cli_session_id
                            let mut sessions = active_sessions.write().await;
                            if let Some(active) = sessions.get_mut(&session_id) {
                                active.cli_session_id = Some(cli_sid.clone());
                                active.last_activity = Instant::now();
                            }
                        }

                        // Collect assistant text for memory
                        if let Message::Assistant {
                            message: ref am, ..
                        } = msg
                        {
                            for block in &am.content {
                                if let ContentBlock::Text(t) = block {
                                    assistant_text_parts.push(t.text.clone());
                                }
                            }
                        }

                        // Convert to ChatEvent(s) and emit + persist structured events
                        let events = Self::message_to_events(msg);
                        for event in events {
                            // Deduplicate ToolUse events — ContentBlockStart and
                            // AssistantMessage can both produce the same tool_use.
                            //
                            // ContentBlockStart arrives first with input: {} (empty),
                            // AssistantMessage arrives later with the FULL input params.
                            //
                            // Strategy:
                            // 1. First occurrence (ContentBlockStart): emit + persist normally
                            // 2. Second occurrence (AssistantMessage): DON'T re-emit to broadcast
                            //    (clients already have the tool_use), but UPDATE the persisted
                            //    record and streaming_events with the full input.
                            if let ChatEvent::ToolUse {
                                ref id, ref input, ..
                            } = event
                            {
                                if let Some(persist_idx) = emitted_tool_use_ids.get(id) {
                                    // Duplicate — update persisted record with full input
                                    let has_real_input = input.is_object()
                                        && input.as_object().is_some_and(|o| !o.is_empty());
                                    if has_real_input {
                                        if let Some(idx) = persist_idx {
                                            if let Some(record) = events_to_persist.get_mut(*idx) {
                                                record.data = serde_json::to_string(&event)
                                                    .unwrap_or_default();
                                                debug!(
                                                    "Updated persisted ToolUse input for id={}",
                                                    id
                                                );
                                            }
                                        }
                                        // Also update in streaming_events snapshot
                                        let mut se = streaming_events.lock().await;
                                        if let Some(existing) = se.iter_mut().find(|e| {
                                            matches!(e, ChatEvent::ToolUse { id: ref eid, .. } if eid == id)
                                        }) {
                                            *existing = event.clone();
                                        }
                                        // Emit ToolUseInputResolved so the frontend can
                                        // update the existing tool_use block's input
                                        emit_chat(
                                            ChatEvent::ToolUseInputResolved {
                                                id: id.clone(),
                                                input: input.clone(),
                                            },
                                            &events_tx,
                                            &nats,
                                            &session_id,
                                        );
                                    }
                                    debug!("Skipping duplicate ToolUse broadcast (id={}), sent input_resolved", id);
                                    continue;
                                }
                                // First occurrence — record it
                                emitted_tool_use_ids.insert(id.clone(), None);
                            }

                            // Persist structured events (skip transient: stream_delta, streaming_status)
                            if !matches!(
                                event,
                                ChatEvent::StreamDelta { .. } | ChatEvent::StreamingStatus { .. }
                            ) {
                                if let Some(uuid) = session_uuid {
                                    let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                                    let persist_idx = events_to_persist.len();
                                    events_to_persist.push(ChatEventRecord {
                                        id: Uuid::new_v4(),
                                        session_id: uuid,
                                        seq,
                                        event_type: event.event_type().to_string(),
                                        data: serde_json::to_string(&event).unwrap_or_default(),
                                        created_at: chrono::Utc::now(),
                                    });
                                    // Track persist index for ToolUse so we can update later
                                    if let ChatEvent::ToolUse { ref id, .. } = event {
                                        emitted_tool_use_ids.insert(id.clone(), Some(persist_idx));
                                    }
                                }
                            }

                            // Accumulate structured events for mid-stream join snapshot.
                            // Excluded:
                            // - StreamDelta: text is in streaming_text (sent as partial_text)
                            // - StreamingStatus: transient, sent explicitly in Phase 1.5
                            // - AssistantText: duplicates streaming_text content (sent as partial_text)
                            if !matches!(
                                event,
                                ChatEvent::StreamDelta { .. }
                                    | ChatEvent::StreamingStatus { .. }
                                    | ChatEvent::AssistantText { .. }
                            ) {
                                // Flush accumulated text before non-text events (ToolUse,
                                // ToolResult, etc.) so the snapshot preserves correct ordering.
                                // Without this, partial_text would contain "Text A + Text B"
                                // with no way to know Text A came before ToolUse.
                                // After flushing, partial_text only contains text streamed
                                // AFTER the last structured event.
                                {
                                    let mut st = streaming_text.lock().await;
                                    if !st.is_empty() {
                                        streaming_events.lock().await.push(
                                            ChatEvent::AssistantText {
                                                content: st.clone(),
                                            },
                                        );
                                        st.clear();
                                    }
                                }
                                streaming_events.lock().await.push(event.clone());
                            }

                            emit_chat(event, &events_tx, &nats, &session_id);
                        }
                    }
                    Err(e) => {
                        error!("Stream error for session {}: {}", session_id, e);
                        emit_chat(
                            ChatEvent::Error {
                                message: format!("Error: {}", e),
                            },
                            &events_tx,
                            &nats,
                            &session_id,
                        );
                        break;
                    }
                }
            }
        } // client lock released here

        // Batch-persist all collected events to Neo4j
        if let Some(uuid) = session_uuid {
            if !events_to_persist.is_empty() {
                if let Err(e) = graph.store_chat_events(uuid, events_to_persist).await {
                    warn!(
                        "Failed to persist {} chat events for session {}: {}",
                        0, session_id, e
                    );
                }
            }
        }

        // Record assistant message and persist to memory store
        if let Some(ref mm) = memory_manager {
            let assistant_text = assistant_text_parts.join("");
            if !assistant_text.is_empty() {
                let mut mm = mm.lock().await;
                mm.record_assistant_message(&assistant_text);

                // Store pending messages via ContextInjector
                if let Some(ref injector) = context_injector {
                    let pending = mm.take_pending_messages();
                    if !pending.is_empty() {
                        if let Err(e) = injector.store_messages(&pending).await {
                            warn!("Failed to store messages for session {}: {}", session_id, e);
                        } else {
                            debug!(
                                "Stored {} messages for session {}",
                                pending.len(),
                                session_id
                            );
                        }
                    }
                }
            }
        }

        // If interrupted, send the interrupt signal to the CLI
        if interrupt_flag.load(Ordering::SeqCst) {
            debug!("Sending interrupt signal to CLI for session {}", session_id);
            let mut c = client.lock().await;
            if let Err(e) = c.interrupt().await {
                warn!(
                    "Failed to send interrupt signal to CLI for session {}: {}",
                    session_id, e
                );
            }
        }

        is_streaming.store(false, Ordering::SeqCst);
        // Broadcast streaming_status=false to all connected clients (multi-tab support)
        emit_chat(
            ChatEvent::StreamingStatus {
                is_streaming: false,
            },
            &events_tx,
            &nats,
            &session_id,
        );
        // Emit CRUD event for session list live refresh
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
        debug!("Stream completed for session {}", session_id);

        // Check pending_messages queue — if there are queued messages, process the next one
        let next_message = {
            let mut queue = pending_messages.lock().await;
            queue.pop_front()
        };

        if let Some(next_msg) = next_message {
            info!(
                "Processing queued message for session {} (queue was non-empty after stream)",
                session_id
            );

            // Persist the queued user_message event and update message count
            if let Some(uuid) = session_uuid {
                // Update message count
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

                let user_event = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: "user_message".to_string(),
                    data: serde_json::to_string(&serde_json::json!({"content": &next_msg}))
                        .unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                };
                let _ = graph.store_chat_events(uuid, vec![user_event]).await;
            }

            // Emit user_message on broadcast + NATS
            emit_chat(
                ChatEvent::UserMessage {
                    content: next_msg.clone(),
                },
                &events_tx,
                &nats,
                &session_id,
            );

            // Recursive call to process the queued message
            // Use Box::pin to handle recursive async
            Box::pin(Self::stream_response(
                client,
                events_tx,
                next_msg,
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
            ))
            .await;
        }
    }

    /// Try to send a message to a session via NATS RPC (cross-instance proxy).
    ///
    /// Returns `Ok(true)` if the message was successfully proxied to the owning instance.
    /// Returns `Ok(false)` if NATS is not configured, no instance responded (timeout),
    /// or the remote instance reported the session is not active there.
    ///
    /// This method has NO side-effects on the local ChatManager — it only communicates
    /// with remote instances via NATS request/reply.
    ///
    /// Callers should fall back to `resume_session()` when this returns `Ok(false)`.
    pub async fn try_remote_send(
        &self,
        session_id: &str,
        message: &str,
        message_type: &str,
    ) -> Result<bool> {
        let Some(ref nats) = self.nats else {
            debug!(
                session_id = %session_id,
                "No NATS configured, skipping remote send"
            );
            return Ok(false);
        };

        info!(
            session_id = %session_id,
            "Attempting NATS RPC send to remote instance"
        );

        match nats
            .request_send_message(session_id, message, message_type)
            .await
        {
            Some(response) if response.success => {
                info!(
                    session_id = %session_id,
                    "Message proxied to remote instance via NATS RPC"
                );
                Ok(true)
            }
            Some(response) => {
                debug!(
                    session_id = %session_id,
                    error = ?response.error,
                    "Remote instance rejected message (session not active there)"
                );
                Ok(false)
            }
            None => {
                debug!(
                    session_id = %session_id,
                    "No NATS RPC reply (timeout) — no instance owns this session"
                );
                Ok(false)
            }
        }
    }

    /// Send a follow-up message to an existing session
    pub async fn send_message(&self, session_id: &str, message: &str) -> Result<()> {
        // Get session state — NO broadcast replacement, the same channel is reused
        let (
            client,
            events_tx,
            interrupt_flag,
            memory_manager,
            next_seq,
            pending_messages,
            is_streaming,
            streaming_text,
            streaming_events,
        ) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

            // Reset interrupt flag for new message
            session.interrupt_flag.store(false, Ordering::SeqCst);
            session.last_activity = Instant::now();

            (
                session.client.clone(),
                session.events_tx.clone(),
                session.interrupt_flag.clone(),
                session.memory_manager.clone(),
                session.next_seq.clone(),
                session.pending_messages.clone(),
                session.is_streaming.clone(),
                session.streaming_text.clone(),
                session.streaming_events.clone(),
            )
        };

        // If a stream is in progress, queue the message for later processing.
        // Do NOT persist or broadcast yet — the dequeue in stream_response() handles that,
        // ensuring the user_message seq comes AFTER the current stream's events.
        if is_streaming.load(Ordering::SeqCst) {
            info!(
                "Stream in progress for session {}, queuing message",
                session_id
            );
            let mut queue = pending_messages.lock().await;
            queue.push_back(message.to_string());
            return Ok(());
        }

        // No stream in progress — persist and broadcast immediately

        // Update message count in Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Ok(Some(node)) = self.graph.get_chat_session(uuid).await {
                let _ = self
                    .graph
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

        // Persist the user_message event
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            let user_event = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: uuid,
                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                event_type: "user_message".to_string(),
                data: serde_json::to_string(&serde_json::json!({"content": message}))
                    .unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            let _ = self.graph.store_chat_events(uuid, vec![user_event]).await;
        }

        // Emit user_message on local broadcast + NATS (visible to all clients)
        let user_msg_event = ChatEvent::UserMessage {
            content: message.to_string(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, user_msg_event);
        }

        // Start streaming directly
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
            )
            .await;
        });

        Ok(())
    }

    /// Resume a previously inactive session by creating a new InteractiveClient.
    ///
    /// If the session has a `cli_session_id`, resumes with `--resume`.
    /// If not (first message or previous spawn failed), starts fresh without `--resume`.
    pub async fn resume_session(&self, session_id: &str, message: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Load session from Neo4j
        let session_node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found in database", session_id))?;

        let cli_session_id = session_node.cli_session_id.as_deref();

        if let Some(cli_id) = cli_session_id {
            info!("Resuming session {} with CLI ID {}", session_id, cli_id);
        } else {
            info!(
                "Starting fresh CLI for session {} (no previous cli_session_id)",
                session_id
            );
        }

        // Build options - with resume flag only if we have a cli_session_id
        let system_prompt = self
            .build_system_prompt(session_node.project_slug.as_deref(), message)
            .await;
        let options = self.build_options(
            &session_node.cwd,
            &session_node.model,
            &system_prompt,
            cli_session_id,
        );

        // Create new InteractiveClient with --resume
        let mut client = InteractiveClient::new(options)
            .map_err(|e| anyhow!("Failed to create InteractiveClient for resume: {}", e))?;

        client
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect resumed InteractiveClient: {}", e))?;

        // Create broadcast channel
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);
        let client = Arc::new(Mutex::new(client));

        // Re-create ConversationMemoryManager for resumed session
        // (uses existing conversation_id if available)
        let memory_manager = if let Some(ref mem_config) = self.memory_config {
            let mm = if let Some(ref conv_id) = session_node.conversation_id {
                ConversationMemoryManager::new(mem_config.clone())
                    .with_conversation_id(conv_id.clone())
            } else {
                let mm = ConversationMemoryManager::new(mem_config.clone());
                // Persist the new conversation_id
                let _ = self
                    .graph
                    .update_chat_session(
                        uuid,
                        None,
                        None,
                        None,
                        None,
                        Some(mm.conversation_id().to_string()),
                        None,
                    )
                    .await;
                mm
            };
            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        // Initialize next_seq from Neo4j (resume existing event history)
        let latest_seq = self
            .graph
            .get_latest_chat_event_seq(uuid)
            .await
            .unwrap_or(0);
        let next_seq = Arc::new(AtomicI64::new(latest_seq + 1));
        let pending_messages = Arc::new(Mutex::new(VecDeque::new()));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let streaming_text = Arc::new(Mutex::new(String::new()));
        let streaming_events = Arc::new(Mutex::new(Vec::new()));

        // Register as active
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            let interrupt_flag = Arc::new(AtomicBool::new(false));
            sessions.insert(
                session_id.to_string(),
                ActiveSession {
                    events_tx: events_tx.clone(),
                    last_activity: Instant::now(),
                    cli_session_id: cli_session_id.map(|s| s.to_string()),
                    client: client.clone(),
                    interrupt_flag: interrupt_flag.clone(),
                    memory_manager: memory_manager.clone(),
                    next_seq: next_seq.clone(),
                    pending_messages: pending_messages.clone(),
                    is_streaming: is_streaming.clone(),
                    streaming_text: streaming_text.clone(),
                    streaming_events: streaming_events.clone(),
                },
            );
            interrupt_flag
        };

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            session_id,
            interrupt_flag.clone(),
            self.active_sessions.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(session_id, self.active_sessions.clone());

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(session_id, self.active_sessions.clone());

        // Persist the user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&serde_json::json!({"content": message}))
                .unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let _ = self.graph.store_chat_events(uuid, vec![user_event]).await;

        // Emit user_message on local broadcast + NATS
        let user_msg_event = ChatEvent::UserMessage {
            content: message.to_string(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, user_msg_event);
        }

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
            )
            .await;
        });

        Ok(())
    }

    /// Retrieve full event history for a session (including tool_use, tool_result, etc.).
    ///
    /// Reads structured `ChatEventRecord` from Neo4j (the same data the WebSocket
    /// replay uses), so every event type is preserved — not just text.
    ///
    /// Falls back to Meilisearch `nexus_messages` for pre-migration sessions that
    /// have no ChatEvent nodes yet (text only, no tool_use).
    pub async fn get_session_messages(
        &self,
        session_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<ChatEventPage> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Verify session exists
        self.graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        let limit_val = limit.unwrap_or(200) as i64;
        let offset_val = offset.unwrap_or(0) as i64;

        // Try Neo4j ChatEvent nodes first (new format — has tool_use, etc.)
        let total_count = self.graph.count_chat_events(uuid).await?;

        if total_count > 0 {
            let events = self
                .graph
                .get_chat_events_paginated(uuid, offset_val, limit_val)
                .await?;

            let has_more = offset_val + events.len() as i64 > total_count;

            return Ok(ChatEventPage {
                events,
                total_count: total_count as usize,
                has_more,
                offset: offset_val as usize,
                limit: limit_val as usize,
            });
        }

        // Fallback: pre-migration sessions — read from Meilisearch (text only)
        let session_node = self.graph.get_chat_session(uuid).await?.unwrap();
        if let Some(conversation_id) = session_node.conversation_id {
            let meili_client = meilisearch_sdk::client::Client::new(
                &self.config.meilisearch_url,
                Some(&self.config.meilisearch_key),
            )
            .map_err(|e| anyhow!("Failed to create Meilisearch client: {}", e))?;

            let index = meili_client.index("nexus_messages");
            let filter = format!("conversation_id = \"{}\"", conversation_id);

            let results: meilisearch_sdk::search::SearchResults<
                nexus_claude::memory::MessageDocument,
            > = index
                .search()
                .with_query("")
                .with_filter(&filter)
                .with_sort(&["created_at:asc"])
                .with_limit(limit_val as usize)
                .with_offset(offset_val as usize)
                .execute()
                .await
                .map_err(|e| anyhow!("Meilisearch query failed: {}", e))?;

            let meili_total = results.estimated_total_hits.unwrap_or(0);

            // Convert MessageDocument → ChatEventRecord (text-only approximation)
            let events: Vec<ChatEventRecord> = results
                .hits
                .into_iter()
                .enumerate()
                .map(|(i, hit)| {
                    let msg = hit.result;
                    let event_type = if msg.role == "user" {
                        "user_message"
                    } else {
                        "assistant_text"
                    };
                    let chat_event = if msg.role == "user" {
                        ChatEvent::UserMessage {
                            content: msg.content.clone(),
                        }
                    } else {
                        ChatEvent::AssistantText {
                            content: msg.content.clone(),
                        }
                    };
                    ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq: (offset_val + i as i64 + 1),
                        event_type: event_type.to_string(),
                        data: serde_json::to_string(&chat_event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    }
                })
                .collect();

            let has_more = offset_val as usize + events.len() < meili_total;

            return Ok(ChatEventPage {
                events,
                total_count: meili_total,
                has_more,
                offset: offset_val as usize,
                limit: limit_val as usize,
            });
        }

        // No events anywhere
        Ok(ChatEventPage {
            events: vec![],
            total_count: 0,
            has_more: false,
            offset: offset_val as usize,
            limit: limit_val as usize,
        })
    }

    /// Backfill title/preview for sessions that have a conversation_id but no title.
    /// Uses Meilisearch to fetch the first user message from each conversation.
    pub async fn backfill_previews_from_meilisearch(&self) -> Result<usize> {
        let injector = match self.context_injector.as_ref() {
            Some(i) => i,
            None => return Ok(0),
        };

        // Get all sessions without title
        let sessions = self
            .graph
            .list_chat_sessions(None, 200, 0)
            .await
            .context("Failed to list sessions")?;

        let mut count = 0;
        for session in &sessions.0 {
            // Skip sessions that already have a title
            if session.title.is_some() {
                continue;
            }
            // Need a conversation_id to look up messages
            let conv_id = match &session.conversation_id {
                Some(c) => c.clone(),
                None => continue,
            };

            // Load the first user message from Meilisearch
            let loaded = match injector.load_conversation(&conv_id, Some(1), Some(0)).await {
                Ok(l) => l,
                Err(_) => continue,
            };

            let first_user_msg = loaded
                .messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone());

            if let Some(content) = first_user_msg {
                let chars: Vec<char> = content.chars().collect();
                let title = if chars.len() > 80 {
                    format!("{}...", chars[..77].iter().collect::<String>().trim_end())
                } else {
                    content.clone()
                };
                let preview = if chars.len() > 200 {
                    format!("{}...", chars[..197].iter().collect::<String>().trim_end())
                } else {
                    content
                };
                let _ = self
                    .graph
                    .update_chat_session(
                        session.id,
                        None,
                        Some(title),
                        None,
                        None,
                        None,
                        Some(preview),
                    )
                    .await;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Search messages across all sessions via Meilisearch full-text search.
    ///
    /// Queries the `nexus_messages` index directly (bypassing the nexus SDK's
    /// `retrieve_context` which applies token_budget / max_context_items limits
    /// designed for LLM context injection, not UI search).
    pub async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<MessageSearchResult>> {
        use nexus_claude::memory::MessageDocument;
        use std::collections::HashMap as StdHashMap;

        // Build a direct Meilisearch client from our config
        let meili_client = meilisearch_sdk::client::Client::new(
            &self.config.meilisearch_url,
            Some(&self.config.meilisearch_key),
        )
        .map_err(|e| anyhow!("Failed to create Meilisearch client: {}", e))?;

        let index = meili_client.index("nexus_messages");

        // Query Meilisearch directly — no token budget, no max_context_items
        let search_results: meilisearch_sdk::search::SearchResults<MessageDocument> = index
            .search()
            .with_query(query)
            .with_limit(limit * 5) // Fetch extra to allow grouping by session
            .with_show_ranking_score(true)
            .execute()
            .await
            .map_err(|e| anyhow!("Meilisearch search failed: {}", e))?;

        if search_results.hits.is_empty() {
            return Ok(vec![]);
        }

        // Group results by conversation_id
        let mut by_conversation: StdHashMap<String, Vec<MessageSearchHit>> = StdHashMap::new();
        for hit in &search_results.hits {
            let doc = &hit.result;
            let score = hit.ranking_score.unwrap_or(0.0);
            let search_hit = MessageSearchHit {
                message_id: doc.id.clone(),
                role: doc.role.clone(),
                content_snippet: truncate_snippet(&doc.content, 300),
                turn_index: doc.turn_index,
                created_at: doc.created_at,
                score,
            };
            by_conversation
                .entry(doc.conversation_id.clone())
                .or_default()
                .push(search_hit);
        }

        // Resolve conversation_id → session metadata from Neo4j
        let (all_sessions, _) = self.graph.list_chat_sessions(None, 200, 0).await?;
        let session_lookup: StdHashMap<String, &crate::neo4j::models::ChatSessionNode> =
            all_sessions
                .iter()
                .filter_map(|s| s.conversation_id.as_ref().map(|cid| (cid.clone(), s)))
                .collect();

        // Build grouped results
        let mut results: Vec<MessageSearchResult> = Vec::new();
        for (conv_id, hits) in by_conversation {
            let session_info = session_lookup.get(&conv_id);

            // Filter by project_slug if specified
            if let Some(filter_slug) = project_slug {
                if let Some(session) = session_info {
                    if session.project_slug.as_deref() != Some(filter_slug) {
                        continue;
                    }
                } else {
                    continue; // No session found, skip
                }
            }

            let best_score = hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);

            results.push(MessageSearchResult {
                session_id: session_info.map(|s| s.id.to_string()).unwrap_or_default(),
                session_title: session_info.and_then(|s| s.title.clone()),
                session_preview: session_info.and_then(|s| s.preview.clone()),
                project_slug: session_info.and_then(|s| s.project_slug.clone()),
                conversation_id: conv_id,
                hits,
                best_score,
            });
        }

        // Sort by best_score descending
        results.sort_by(|a, b| {
            b.best_score
                .partial_cmp(&a.best_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to requested number of sessions
        results.truncate(limit);

        Ok(results)
    }

    /// Subscribe to a session's broadcast channel (used by WebSocket handler)
    pub async fn subscribe(&self, session_id: &str) -> Result<broadcast::Receiver<ChatEvent>> {
        let sessions = self.active_sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        Ok(session.events_tx.subscribe())
    }

    /// Get persisted events since a given sequence number (for WebSocket replay)
    pub async fn get_events_since(
        &self,
        session_id: &str,
        after_seq: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;
        self.graph
            .get_chat_events(uuid, after_seq, 10000) // generous limit for replay
            .await
    }

    /// Check if a session is currently streaming
    pub async fn is_session_streaming(&self, session_id: &str) -> bool {
        let sessions = self.active_sessions.read().await;
        sessions
            .get(session_id)
            .map(|s| s.is_streaming.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Get a snapshot of the current streaming state for mid-stream join.
    ///
    /// Returns `(is_streaming, accumulated_text, streaming_events)`.
    /// - `accumulated_text`: all stream_delta text since the current stream started
    /// - `streaming_events`: all structured events (ToolUse, ToolResult, AssistantText, etc.)
    ///   since the current stream started, excluding StreamDelta (which is in accumulated_text)
    ///
    /// This allows a newly connected WebSocket client to fully reconstruct the
    /// in-progress assistant turn, including tool calls.
    pub async fn get_streaming_snapshot(&self, session_id: &str) -> (bool, String, Vec<ChatEvent>) {
        let sessions = self.active_sessions.read().await;
        match sessions.get(session_id) {
            Some(session) => {
                let is_streaming = session.is_streaming.load(Ordering::SeqCst);
                let text = session.streaming_text.lock().await.clone();
                let events = session.streaming_events.lock().await.clone();
                (is_streaming, text, events)
            }
            None => (false, String::new(), Vec::new()),
        }
    }

    /// Interrupt the current operation in a session.
    ///
    /// Sets the interrupt flag, which causes the stream loop to break and release the
    /// client lock. The stream loop then sends the actual interrupt signal to the CLI.
    /// This is instantaneous — no waiting for the Mutex.
    pub async fn interrupt(&self, session_id: &str) -> Result<()> {
        let interrupt_flag = {
            let sessions = self.active_sessions.read().await;
            sessions.get(session_id).map(|s| s.interrupt_flag.clone())
        };

        if let Some(flag) = interrupt_flag {
            // Session is local — set the flag so the stream loop breaks
            flag.store(true, Ordering::SeqCst);
            info!(
                session_id = %session_id,
                "Interrupt flag set locally"
            );
        } else {
            debug!(
                session_id = %session_id,
                "Session not active locally, interrupt will be routed via NATS only"
            );
        }

        // Always publish interrupt to NATS so the owning instance (if remote) also stops.
        // This is fire-and-forget — no-op if NATS is not configured.
        if let Some(ref nats) = self.nats {
            nats.publish_interrupt(session_id);
            debug!(
                session_id = %session_id,
                "Interrupt published to NATS"
            );
        }

        Ok(())
    }

    /// Close an active session: disconnect client, remove from active map
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let client = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .remove(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.client
        };

        let mut c = client.lock().await;
        if let Err(e) = c.disconnect().await {
            warn!("Error disconnecting session {}: {}", session_id, e);
        }

        info!("Closed session {}", session_id);
        Ok(())
    }

    /// Start a background task that cleans up timed-out sessions
    pub fn start_cleanup_task(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let timeout = manager.config.session_timeout;
        let interval = timeout / 2; // Check at half the timeout interval

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                let expired: Vec<String> = {
                    let sessions = manager.active_sessions.read().await;
                    sessions
                        .iter()
                        .filter(|(_, s)| s.last_activity.elapsed() > timeout)
                        .map(|(id, _)| id.clone())
                        .collect()
                };

                for id in expired {
                    info!("Cleaning up timed-out session {}", id);
                    if let Err(e) = manager.close_session(&id).await {
                        warn!("Failed to close timed-out session {}: {}", id, e);
                    }
                }
            }
        });
    }

    /// Get the number of currently active sessions
    pub async fn active_session_count(&self) -> usize {
        self.active_sessions.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::ChatSessionNode;
    use crate::test_helpers::{mock_app_state, test_chat_session, test_project};
    use nexus_claude::{
        AssistantMessage, ContentBlock, ContentValue, TextContent, ThinkingContent,
        ToolResultContent, ToolUseContent,
    };
    use std::path::PathBuf;
    use std::time::Duration;

    fn test_config() -> ChatConfig {
        ChatConfig {
            mcp_server_path: PathBuf::from("/usr/bin/mcp_server"),
            default_model: "claude-opus-4-6".into(),
            max_sessions: 10,
            session_timeout: Duration::from_secs(1800),
            neo4j_uri: "bolt://localhost:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "test".into(),
            meilisearch_url: "http://localhost:7700".into(),
            meilisearch_key: "key".into(),
            nats_url: None,
            max_turns: 10,
            prompt_builder_model: "claude-opus-4-6".into(),
        }
    }

    #[test]
    fn test_resolve_model_with_override() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(
            manager.resolve_model(Some("claude-sonnet-4-20250514")),
            "claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_resolve_model_default() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(manager.resolve_model(None), "claude-opus-4-6");
    }

    #[tokio::test]
    async fn test_build_system_prompt_no_project() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let prompt = manager.build_system_prompt(None, "test").await;
        assert!(prompt.contains("Project Orchestrator"));
        assert!(prompt.contains("EXCLUSIVEMENT les outils MCP"));
        assert!(!prompt.contains("Projet actif"));
    }

    #[tokio::test]
    async fn test_build_system_prompt_with_project() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let prompt = manager
            .build_system_prompt(Some(&project.slug), "help me plan")
            .await;

        // Contains the base prompt
        assert!(prompt.contains("EXCLUSIVEMENT les outils MCP"));
        // Contains dynamic context section (either oneshot or fallback)
        assert!(prompt.contains("---"));
        // The project name should appear somewhere in the dynamic context
        assert!(prompt.contains(&project.name));
    }

    #[tokio::test]
    async fn test_session_not_active_by_default() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert!(!manager.is_session_active("nonexistent").await);
    }

    // ====================================================================
    // build_options
    // ====================================================================

    #[test]
    fn test_build_options_basic() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager.build_options(
            "/tmp/project",
            "claude-opus-4-6",
            "System prompt here",
            None,
        );

        assert_eq!(options.model, Some("claude-opus-4-6".into()));
        assert_eq!(options.cwd, Some(PathBuf::from("/tmp/project")));
        assert!(options.resume.is_none());
        assert!(options.mcp_servers.contains_key("project-orchestrator"));
    }

    #[test]
    fn test_build_options_with_resume() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager.build_options(
            "/tmp/project",
            "claude-opus-4-6",
            "System prompt",
            Some("cli-session-abc"),
        );

        assert_eq!(options.resume, Some("cli-session-abc".into()));
    }

    #[test]
    fn test_build_options_mcp_server_config() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let options = manager.build_options("/tmp", "model", "prompt", None);

        let mcp = options.mcp_servers.get("project-orchestrator").unwrap();
        match mcp {
            McpServerConfig::Stdio { command, env, .. } => {
                assert_eq!(command, "/usr/bin/mcp_server");
                let env = env.as_ref().unwrap();
                assert_eq!(env.get("NEO4J_URI").unwrap(), "bolt://localhost:7687");
                assert_eq!(env.get("NEO4J_USER").unwrap(), "neo4j");
                assert_eq!(env.get("NEO4J_PASSWORD").unwrap(), "test");
                assert_eq!(env.get("MEILISEARCH_URL").unwrap(), "http://localhost:7700");
                assert_eq!(env.get("MEILISEARCH_KEY").unwrap(), "key");
            }
            _ => panic!("Expected Stdio MCP config"),
        }
    }

    // ====================================================================
    // message_to_events
    // ====================================================================

    #[test]
    fn test_message_to_events_assistant_text() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Hello!".into(),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::AssistantText { content } if content == "Hello!"));
    }

    #[test]
    fn test_message_to_events_thinking() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::Thinking(ThinkingContent {
                    thinking: "Let me think...".into(),
                    signature: "sig".into(),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::Thinking { content } if content == "Let me think...")
        );
    }

    #[test]
    fn test_message_to_events_tool_use() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolUse(ToolUseContent {
                    id: "tool-1".into(),
                    name: "create_plan".into(),
                    input: serde_json::json!({"title": "My Plan"}),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
            if id == "tool-1" && tool == "create_plan"));
    }

    #[test]
    fn test_message_to_events_tool_result() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-1".into(),
                    content: Some(ContentValue::Text("Success".into())),
                    is_error: Some(false),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::ToolResult { id, is_error, .. }
            if id == "tool-1" && !is_error)
        );
    }

    #[test]
    fn test_message_to_events_tool_result_error() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-2".into(),
                    content: Some(ContentValue::Text("Not found".into())),
                    is_error: Some(true),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolResult { is_error, .. } if *is_error));
    }

    #[test]
    fn test_message_to_events_result() {
        let msg = Message::Result {
            subtype: "success".into(),
            duration_ms: 5000,
            duration_api_ms: 4500,
            is_error: false,
            num_turns: 3,
            session_id: "cli-abc-123".into(),
            total_cost_usd: Some(0.15),
            usage: None,
            result: None,
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            session_id, duration_ms, cost_usd
        } if session_id == "cli-abc-123" && *duration_ms == 5000 && *cost_usd == Some(0.15)));
    }

    #[test]
    fn test_message_to_events_multiple_blocks() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(ThinkingContent {
                        thinking: "hmm...".into(),
                        signature: "s".into(),
                    }),
                    ContentBlock::Text(TextContent {
                        text: "Here is my answer".into(),
                    }),
                    ContentBlock::ToolUse(ToolUseContent {
                        id: "t1".into(),
                        name: "list_plans".into(),
                        input: serde_json::json!({}),
                    }),
                ],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ChatEvent::Thinking { .. }));
        assert!(matches!(&events[1], ChatEvent::AssistantText { .. }));
        assert!(matches!(&events[2], ChatEvent::ToolUse { .. }));
    }

    #[test]
    fn test_message_to_events_system_message() {
        let msg = Message::System {
            subtype: "init".into(),
            data: serde_json::json!({"version": "1.0"}),
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_user_message() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: "Hi".into(),
                content_blocks: None,
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_user_message_with_tool_result() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: String::new(),
                content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "toolu_abc123".into(),
                    content: Some(ContentValue::Text("fn main() {}".into())),
                    is_error: Some(false),
                })]),
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ChatEvent::ToolResult { id, is_error, .. }
            if id == "toolu_abc123" && !is_error
        ));
    }

    #[test]
    fn test_message_to_events_user_message_with_tool_result_error() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: String::new(),
                content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "toolu_err001".into(),
                    content: Some(ContentValue::Text("File not found".into())),
                    is_error: Some(true),
                })]),
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ChatEvent::ToolResult { id, is_error, .. }
            if id == "toolu_err001" && *is_error
        ));
    }

    // ====================================================================
    // message_to_events — StreamEvent
    // ====================================================================

    #[test]
    fn test_message_to_events_stream_text_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::TextDelta {
                    text: "Hello".into(),
                },
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::StreamDelta { text } if text == "Hello"));
    }

    #[test]
    fn test_message_to_events_stream_content_block_start_tool_use() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 1,
                content_block: serde_json::json!({
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "list_plans",
                    "input": {"status": "active"}
                }),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
                if id == "toolu_abc123" && tool == "list_plans"));
    }

    #[test]
    fn test_message_to_events_stream_content_block_start_text_ignored() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 0,
                content_block: serde_json::json!({"type": "text", "text": ""}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_thinking_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::ThinkingDelta {
                    thinking: "hmm".into(),
                },
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_stop() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageStop,
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_content_block_start() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 0,
                content_block: serde_json::json!({"type": "text", "text": ""}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_input_json_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::InputJsonDelta {
                    partial_json: r#"{"title":"#.into(),
                },
            },
            session_id: Some("sess-1".into()),
            parent_tool_use_id: None,
        };

        // InputJsonDelta is not TextDelta, so it should produce empty events
        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_start() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageStart {
                message: serde_json::json!({"id": "msg_123"}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageDelta {
                delta: serde_json::json!({"stop_reason": "end_turn"}),
                usage: Some(serde_json::json!({"output_tokens": 50})),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_tool_result_structured() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-3".into(),
                    content: Some(ContentValue::Structured(vec![
                        serde_json::json!({"type": "text", "text": "result 1"}),
                        serde_json::json!({"type": "text", "text": "result 2"}),
                    ])),
                    is_error: None,
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolResult {
                id,
                result,
                is_error,
            } => {
                assert_eq!(id, "tool-3");
                assert!(result.is_array());
                assert_eq!(result.as_array().unwrap().len(), 2);
                assert!(!is_error); // None defaults to false
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_message_to_events_tool_result_none_content() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-4".into(),
                    content: None,
                    is_error: Some(false),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolResult { result, .. } => {
                assert!(result.is_null());
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    // ====================================================================
    // build_system_prompt with active plans
    // ====================================================================

    #[tokio::test]
    async fn test_build_system_prompt_with_active_plans() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        // Create an active plan linked to the project
        let plan = crate::test_helpers::test_plan();
        state.neo4j.create_plan(&plan).await.unwrap();
        state
            .neo4j
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let prompt = manager
            .build_system_prompt(Some(&project.slug), "check the plan")
            .await;

        // Base prompt present
        assert!(prompt.contains("EXCLUSIVEMENT les outils MCP"));
        // Dynamic context section present (either oneshot or fallback)
        assert!(prompt.contains("---"));
        // Project name should appear in the dynamic context
        assert!(prompt.contains(&project.name));
    }

    // ====================================================================
    // active_session_count
    // ====================================================================

    #[tokio::test]
    async fn test_active_session_count_empty() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(manager.active_session_count().await, 0);
    }

    // ====================================================================
    // subscribe / interrupt / close errors for missing sessions
    // ====================================================================

    #[tokio::test]
    async fn test_subscribe_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.subscribe("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_interrupt_nonexistent_session() {
        // interrupt() no longer errors for non-local sessions — it always succeeds
        // because it may need to publish to NATS for cross-instance interrupt routing.
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent").await;
        assert!(
            result.is_ok(),
            "interrupt should succeed even for non-local sessions"
        );
    }

    #[tokio::test]
    async fn test_close_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.close_session("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_message_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.send_message("nonexistent", "hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resume_session_invalid_uuid() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.resume_session("not-a-uuid", "hello").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid session ID"));
    }

    #[tokio::test]
    async fn test_resume_session_not_in_db() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let id = Uuid::new_v4().to_string();
        let result = manager.resume_session(&id, "hello").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in database"));
    }

    #[tokio::test]
    async fn test_resume_session_no_cli_session_id_starts_fresh() {
        // When a session has no cli_session_id (first message or previous spawn failed),
        // resume_session should attempt to start a fresh CLI (not error immediately).
        // In CI without Claude CLI, this will fail at InteractiveClient creation,
        // but the error should NOT be "no CLI session ID".
        let state = mock_app_state();
        let session = test_chat_session(None); // no cli_session_id
        state.neo4j.create_chat_session(&session).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .resume_session(&session.id.to_string(), "hello")
            .await;
        // Should fail (CLI not available in test), but NOT with "no CLI session ID"
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("no CLI session ID"),
            "Should not fail with 'no CLI session ID', got: {}",
            err_msg
        );
    }

    // ====================================================================
    // ChatSession CRUD via GraphStore (mock)
    // ====================================================================

    #[tokio::test]
    async fn test_chat_session_crud_lifecycle() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        // Create
        let session = test_chat_session(Some("my-project"));
        graph.create_chat_session(&session).await.unwrap();

        // Get
        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.id, session.id);
        assert_eq!(fetched.cwd, "/tmp/test");
        assert_eq!(fetched.model, "claude-opus-4-6");
        assert_eq!(fetched.project_slug.as_deref(), Some("my-project"));

        // Update
        let updated = graph
            .update_chat_session(
                session.id,
                Some("cli-abc-123".into()),
                Some("My Chat".into()),
                Some(5),
                Some(0.25),
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.cli_session_id.as_deref(), Some("cli-abc-123"));
        assert_eq!(updated.title.as_deref(), Some("My Chat"));
        assert_eq!(updated.message_count, 5);
        assert_eq!(updated.total_cost_usd, Some(0.25));

        // Delete
        let deleted = graph.delete_chat_session(session.id).await.unwrap();
        assert!(deleted);

        // Get after delete
        let gone = graph.get_chat_session(session.id).await.unwrap();
        assert!(gone.is_none());
    }

    #[tokio::test]
    async fn test_chat_session_list_with_filter() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let s1 = test_chat_session(Some("project-a"));
        let s2 = test_chat_session(Some("project-a"));
        let s3 = test_chat_session(Some("project-b"));
        let s4 = test_chat_session(None);

        graph.create_chat_session(&s1).await.unwrap();
        graph.create_chat_session(&s2).await.unwrap();
        graph.create_chat_session(&s3).await.unwrap();
        graph.create_chat_session(&s4).await.unwrap();

        // All sessions
        let (all, total) = graph.list_chat_sessions(None, 50, 0).await.unwrap();
        assert_eq!(total, 4);
        assert_eq!(all.len(), 4);

        // Filter by project-a
        let (filtered, total) = graph
            .list_chat_sessions(Some("project-a"), 50, 0)
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(filtered.len(), 2);

        // Pagination
        let (page, total) = graph
            .list_chat_sessions(Some("project-a"), 1, 0)
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(page.len(), 1);
    }

    #[tokio::test]
    async fn test_chat_session_update_partial() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();

        // Update only title
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                Some("Title only".into()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.title.as_deref(), Some("Title only"));
        assert!(updated.cli_session_id.is_none()); // unchanged
        assert_eq!(updated.message_count, 0); // unchanged
    }

    #[tokio::test]
    async fn test_chat_session_update_nonexistent() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let result = graph
            .update_chat_session(uuid::Uuid::new_v4(), None, None, None, None, None, None)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_chat_session_delete_nonexistent() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let deleted = graph
            .delete_chat_session(uuid::Uuid::new_v4())
            .await
            .unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization() {
        let session = ChatSessionNode {
            id: uuid::Uuid::new_v4(),
            cli_session_id: Some("cli-123".into()),
            project_slug: Some("test-proj".into()),
            cwd: "/home/user/code".into(),
            title: Some("My session".into()),
            model: "claude-opus-4-6".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 10,
            total_cost_usd: Some(1.50),
            conversation_id: Some("conv-abc-123".into()),
            preview: Some("Hello, can you help me with this?".into()),
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.cli_session_id, session.cli_session_id);
        assert_eq!(deserialized.message_count, 10);
        assert_eq!(deserialized.total_cost_usd, Some(1.50));
    }

    // ====================================================================
    // new_without_memory — fields
    // ====================================================================

    #[test]
    fn test_new_without_memory_has_no_injector() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert!(manager.context_injector.is_none());
        assert!(manager.memory_config.is_none());
    }

    // ====================================================================
    // get_session_messages — error paths
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_messages_no_injector() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Session doesn't exist in mock store — should get "not found"
        let result = manager
            .get_session_messages(&Uuid::new_v4().to_string(), None, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_session_messages_invalid_uuid() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.get_session_messages("not-a-uuid", None, None).await;
        assert!(result.is_err());
        // Invalid UUID is rejected before any storage lookup
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid session ID"));
    }

    // ====================================================================
    // get_session_messages — returns structured events (tool_use, etc.)
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_messages_returns_tool_use_events() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        // Create a session
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Store events including tool_use and tool_result
        let events = vec![
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 1,
                event_type: "user_message".into(),
                data: serde_json::to_string(&ChatEvent::UserMessage {
                    content: "List my plans".into(),
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 2,
                event_type: "tool_use".into(),
                data: serde_json::to_string(&ChatEvent::ToolUse {
                    id: "tu_1".into(),
                    tool: "list_plans".into(),
                    input: serde_json::json!({"status": "in_progress"}),
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 3,
                event_type: "tool_result".into(),
                data: serde_json::to_string(&ChatEvent::ToolResult {
                    id: "tu_1".into(),
                    result: serde_json::json!({"plans": []}),
                    is_error: false,
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 4,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&ChatEvent::AssistantText {
                    content: "You have no in-progress plans.".into(),
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
        ];
        state
            .neo4j
            .store_chat_events(session_id, events)
            .await
            .unwrap();

        // Retrieve via get_session_messages
        let page = manager
            .get_session_messages(&session_id.to_string(), None, None)
            .await
            .unwrap();

        assert_eq!(page.total_count, 4);
        assert_eq!(page.events.len(), 4);

        // Verify event types are preserved
        assert_eq!(page.events[0].event_type, "user_message");
        assert_eq!(page.events[1].event_type, "tool_use");
        assert_eq!(page.events[2].event_type, "tool_result");
        assert_eq!(page.events[3].event_type, "assistant_text");

        // Verify tool_use data is intact
        let tool_use: ChatEvent = serde_json::from_str(&page.events[1].data).unwrap();
        match tool_use {
            ChatEvent::ToolUse { id, tool, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(tool, "list_plans");
                assert_eq!(input, serde_json::json!({"status": "in_progress"}));
            }
            _ => panic!("Expected ToolUse event"),
        }

        // Verify ordering by seq
        assert_eq!(page.events[0].seq, 1);
        assert_eq!(page.events[1].seq, 2);
        assert_eq!(page.events[2].seq, 3);
        assert_eq!(page.events[3].seq, 4);
    }

    #[tokio::test]
    async fn test_get_session_messages_pagination() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Store 5 events
        let events: Vec<ChatEventRecord> = (1..=5)
            .map(|i| ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: i,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&ChatEvent::AssistantText {
                    content: format!("Message {}", i),
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            })
            .collect();
        state
            .neo4j
            .store_chat_events(session_id, events)
            .await
            .unwrap();

        // First page: offset=0, limit=2
        let page1 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(0))
            .await
            .unwrap();
        assert_eq!(page1.events.len(), 2);
        assert_eq!(page1.total_count, 5);
        assert_eq!(page1.events[0].seq, 1);
        assert_eq!(page1.events[1].seq, 2);

        // Second page: offset=2, limit=2
        let page2 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(2))
            .await
            .unwrap();
        assert_eq!(page2.events.len(), 2);
        assert_eq!(page2.events[0].seq, 3);
        assert_eq!(page2.events[1].seq, 4);

        // Last page: offset=4, limit=2
        let page3 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(4))
            .await
            .unwrap();
        assert_eq!(page3.events.len(), 1);
        assert_eq!(page3.events[0].seq, 5);
    }

    #[tokio::test]
    async fn test_get_session_messages_empty_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        // Create session with no events
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let page = manager
            .get_session_messages(&session_id.to_string(), None, None)
            .await
            .unwrap();

        assert_eq!(page.total_count, 0);
        assert!(page.events.is_empty());
        assert!(!page.has_more);
    }

    // ====================================================================
    // conversation_id in mock GraphStore CRUD
    // ====================================================================

    #[tokio::test]
    async fn test_chat_session_update_conversation_id() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();
        assert!(session.conversation_id.is_none());

        // Update conversation_id
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                None,
                None,
                None,
                Some("conv-new-123".into()),
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.conversation_id.as_deref(), Some("conv-new-123"));

        // Fetch and verify persisted
        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.conversation_id.as_deref(), Some("conv-new-123"));
    }

    #[tokio::test]
    async fn test_chat_session_create_with_conversation_id() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let mut session = test_chat_session(Some("proj"));
        session.conversation_id = Some("conv-init-456".into());
        graph.create_chat_session(&session).await.unwrap();

        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.conversation_id.as_deref(), Some("conv-init-456"));
    }

    #[tokio::test]
    async fn test_chat_session_conversation_id_survives_other_updates() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();

        // Set conversation_id
        graph
            .update_chat_session(
                session.id,
                None,
                None,
                None,
                None,
                Some("conv-persist".into()),
                None,
            )
            .await
            .unwrap();

        // Update title only — conversation_id should be preserved
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                Some("New Title".into()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.title.as_deref(), Some("New Title"));
        assert_eq!(updated.conversation_id.as_deref(), Some("conv-persist"));
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization_with_conversation_id() {
        let session = ChatSessionNode {
            id: uuid::Uuid::new_v4(),
            cli_session_id: None,
            project_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "model".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: Some("conv-serde-test".into()),
            preview: None,
        };

        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains("conv-serde-test"));

        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.conversation_id.as_deref(),
            Some("conv-serde-test")
        );
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization_without_conversation_id() {
        let session = ChatSessionNode {
            id: uuid::Uuid::new_v4(),
            cli_session_id: None,
            project_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "model".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.conversation_id.is_none());
    }

    // ====================================================================
    // get_streaming_snapshot
    // ====================================================================

    #[tokio::test]
    async fn test_get_streaming_snapshot_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let (is_streaming, text, events) = manager.get_streaming_snapshot("nonexistent").await;
        assert!(!is_streaming);
        assert!(text.is_empty());
        assert!(events.is_empty());
    }

    /// Helper: create a dummy InteractiveClient for tests.
    /// Returns None if the Claude CLI is not installed (e.g., in CI).
    fn try_create_dummy_client() -> Option<InteractiveClient> {
        let opts = ClaudeCodeOptions {
            model: Some("test".into()),
            ..Default::default()
        };
        InteractiveClient::new(opts).ok()
    }

    /// Helper: create a dummy ActiveSession for testing.
    /// Returns None if the Claude CLI is not installed.
    fn try_create_dummy_session(
        is_streaming: bool,
        streaming_text: &str,
        streaming_events_data: Vec<ChatEvent>,
    ) -> Option<(ActiveSession, Arc<Mutex<VecDeque<String>>>)> {
        let client = try_create_dummy_client()?;
        let (tx, _rx) = broadcast::channel(16);
        let pending_messages = Arc::new(Mutex::new(VecDeque::new()));

        let session = ActiveSession {
            events_tx: tx,
            last_activity: Instant::now(),
            cli_session_id: None,
            client: Arc::new(Mutex::new(client)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            memory_manager: None,
            next_seq: Arc::new(AtomicI64::new(1)),
            pending_messages: pending_messages.clone(),
            is_streaming: Arc::new(AtomicBool::new(is_streaming)),
            streaming_text: Arc::new(Mutex::new(streaming_text.to_string())),
            streaming_events: Arc::new(Mutex::new(streaming_events_data)),
        };

        Some((session, pending_messages))
    }

    #[tokio::test]
    async fn test_get_streaming_snapshot_with_active_session_not_streaming() {
        let Some((session, _)) = try_create_dummy_session(false, "", vec![]) else {
            eprintln!("Skipping test: Claude CLI not installed");
            return;
        };

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        manager
            .active_sessions
            .write()
            .await
            .insert("test-session".into(), session);

        let (is_streaming, text, events) = manager.get_streaming_snapshot("test-session").await;
        assert!(!is_streaming);
        assert!(text.is_empty());
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_get_streaming_snapshot_with_active_streaming_session() {
        let events_data = vec![
            ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "list_plans".into(),
                input: serde_json::json!({}),
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"plans": []}),
                is_error: false,
            },
        ];

        let Some((session, _)) = try_create_dummy_session(true, "Hello world", events_data) else {
            eprintln!("Skipping test: Claude CLI not installed");
            return;
        };

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        manager
            .active_sessions
            .write()
            .await
            .insert("streaming-session".into(), session);

        let (is_streaming, text, events) =
            manager.get_streaming_snapshot("streaming-session").await;
        assert!(is_streaming);
        assert_eq!(text, "Hello world");
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], ChatEvent::ToolUse { tool, .. } if tool == "list_plans"));
        assert!(matches!(&events[1], ChatEvent::ToolResult { id, .. } if id == "t1"));
    }

    // ====================================================================
    // send_message — queuing when is_streaming=true
    // ====================================================================

    #[tokio::test]
    async fn test_send_message_queues_when_streaming() {
        let Some((session, pending_messages)) = try_create_dummy_session(true, "", vec![]) else {
            eprintln!("Skipping test: Claude CLI not installed");
            return;
        };

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let session_id = Uuid::new_v4().to_string();

        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), session);

        // Send a message while streaming — should be queued, NOT sent
        let result = manager.send_message(&session_id, "queued message").await;
        assert!(result.is_ok());

        // Verify the message was queued
        let queue = pending_messages.lock().await;
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0], "queued message");
    }

    #[tokio::test]
    async fn test_send_message_queues_multiple_when_streaming() {
        let Some((session, pending_messages)) = try_create_dummy_session(true, "", vec![]) else {
            eprintln!("Skipping test: Claude CLI not installed");
            return;
        };

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let session_id = Uuid::new_v4().to_string();

        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), session);

        // Queue multiple messages
        manager.send_message(&session_id, "first").await.unwrap();
        manager.send_message(&session_id, "second").await.unwrap();
        manager.send_message(&session_id, "third").await.unwrap();

        let queue = pending_messages.lock().await;
        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0], "first");
        assert_eq!(queue[1], "second");
        assert_eq!(queue[2], "third");
    }

    // ====================================================================
    // streaming_events filtering logic
    // ====================================================================

    #[test]
    fn test_streaming_events_filter_excludes_transient_events() {
        // This tests the filtering logic used in stream_response():
        // StreamDelta, StreamingStatus, and AssistantText should be EXCLUDED
        // from streaming_events buffer (they are handled via streaming_text or
        // sent explicitly in Phase 1.5)

        let events_to_exclude = vec![
            ChatEvent::StreamDelta {
                text: "Hello".into(),
            },
            ChatEvent::StreamingStatus { is_streaming: true },
            ChatEvent::StreamingStatus {
                is_streaming: false,
            },
            ChatEvent::AssistantText {
                content: "Hello world".into(),
            },
        ];

        for event in &events_to_exclude {
            let should_add = !matches!(
                event,
                ChatEvent::StreamDelta { .. }
                    | ChatEvent::StreamingStatus { .. }
                    | ChatEvent::AssistantText { .. }
            );
            assert!(
                !should_add,
                "Event {:?} should be excluded from streaming_events",
                event.event_type()
            );
        }
    }

    #[test]
    fn test_streaming_events_filter_includes_structured_events() {
        // These events SHOULD be included in streaming_events buffer
        // so mid-stream joiners can reconstruct tool calls

        let events_to_include = vec![
            ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "create_plan".into(),
                input: serde_json::json!({"title": "Plan"}),
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
            },
            ChatEvent::PermissionRequest {
                id: "p1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            },
            ChatEvent::InputRequest {
                prompt: "Choose:".into(),
                options: Some(vec!["A".into(), "B".into()]),
            },
            ChatEvent::Error {
                message: "Something went wrong".into(),
            },
            ChatEvent::Result {
                session_id: "cli-123".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
            },
            ChatEvent::UserMessage {
                content: "Hello".into(),
            },
        ];

        for event in &events_to_include {
            let should_add = !matches!(
                event,
                ChatEvent::StreamDelta { .. }
                    | ChatEvent::StreamingStatus { .. }
                    | ChatEvent::AssistantText { .. }
            );
            assert!(
                should_add,
                "Event {:?} should be included in streaming_events",
                event.event_type()
            );
        }
    }

    // ====================================================================
    // ActiveSession with streaming_events field
    // ====================================================================

    #[tokio::test]
    async fn test_active_session_streaming_events_field() {
        let Some((session, _)) = try_create_dummy_session(false, "", vec![]) else {
            eprintln!("Skipping test: Claude CLI not installed");
            return;
        };

        // Verify streaming_events starts empty
        assert!(session.streaming_events.lock().await.is_empty());

        // Push an event and verify it's accessible
        session
            .streaming_events
            .lock()
            .await
            .push(ChatEvent::Thinking {
                content: "test".into(),
            });
        assert_eq!(session.streaming_events.lock().await.len(), 1);

        // Clear and verify
        session.streaming_events.lock().await.clear();
        assert!(session.streaming_events.lock().await.is_empty());
    }

    // ====================================================================
    // ChatEvent event_type() — covers the StreamingStatus variant
    // ====================================================================

    #[test]
    fn test_chat_event_streaming_status_type() {
        let event = ChatEvent::StreamingStatus { is_streaming: true };
        assert_eq!(event.event_type(), "streaming_status");

        let event = ChatEvent::StreamingStatus {
            is_streaming: false,
        };
        assert_eq!(event.event_type(), "streaming_status");
    }

    #[test]
    fn test_chat_event_user_message_type() {
        let event = ChatEvent::UserMessage {
            content: "Hello".into(),
        };
        assert_eq!(event.event_type(), "user_message");
    }

    #[test]
    fn test_chat_event_streaming_status_serde_roundtrip() {
        let event = ChatEvent::StreamingStatus { is_streaming: true };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"streaming_status\""));
        assert!(json.contains("\"is_streaming\":true"));

        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            deserialized,
            ChatEvent::StreamingStatus { is_streaming: true }
        ));
    }

    #[test]
    fn test_chat_event_user_message_serde_roundtrip() {
        let event = ChatEvent::UserMessage {
            content: "Hello world".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"user_message\""));

        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(deserialized, ChatEvent::UserMessage { ref content } if content == "Hello world")
        );
    }

    // ====================================================================
    // with_event_emitter builder
    // ====================================================================

    #[test]
    fn test_with_event_emitter_sets_emitter() {
        use crate::events::{CrudEvent, EventEmitter};

        struct DummyEmitter;
        impl EventEmitter for DummyEmitter {
            fn emit(&self, _event: CrudEvent) {}
        }

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        assert!(manager.event_emitter.is_none());

        let manager = manager.with_event_emitter(Arc::new(DummyEmitter));
        assert!(manager.event_emitter.is_some());
    }

    // ====================================================================
    // backfill_previews_from_meilisearch — early return when no injector
    // ====================================================================

    #[tokio::test]
    async fn test_backfill_previews_no_injector_returns_zero() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        // new_without_memory => context_injector is None => should return Ok(0)
        let result = manager.backfill_previews_from_meilisearch().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    // ====================================================================
    // update_chat_session via mock — title, preview, conversation_id
    // ====================================================================

    #[tokio::test]
    async fn test_update_session_title_and_preview() {
        let state = mock_app_state();
        let session = test_chat_session(Some("test-proj"));
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Update title
        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                None,
                Some("My new title".into()),
                None,
                None,
                None,
                Some("Preview of the conversation".into()),
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.title.as_deref(), Some("My new title"));
        assert_eq!(s.preview.as_deref(), Some("Preview of the conversation"));
    }

    #[tokio::test]
    async fn test_update_session_conversation_id() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                None,
                None,
                None,
                None,
                Some("conv-abc-123".into()),
                None,
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        assert_eq!(
            updated.unwrap().conversation_id.as_deref(),
            Some("conv-abc-123")
        );
    }

    #[tokio::test]
    async fn test_update_session_cost_and_message_count() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(session_id, None, None, Some(42), Some(1.50), None, None)
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.message_count, 42);
        assert!((s.total_cost_usd.unwrap() - 1.50).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_update_session_not_found() {
        let state = mock_app_state();
        let fake_id = Uuid::new_v4();
        let updated = state
            .neo4j
            .update_chat_session(fake_id, None, None, None, None, None, None)
            .await
            .unwrap();
        assert!(updated.is_none());
    }

    #[tokio::test]
    async fn test_update_session_cli_session_id() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                Some("cli-session-xyz".into()),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        assert_eq!(
            updated.unwrap().cli_session_id.as_deref(),
            Some("cli-session-xyz")
        );
    }

    #[tokio::test]
    async fn test_update_session_partial_preserves_existing() {
        let state = mock_app_state();
        let mut session = test_chat_session(Some("my-proj"));
        session.title = Some("Original title".into());
        session.preview = Some("Original preview".into());
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Update only message_count — should preserve title and preview
        let updated = state
            .neo4j
            .update_chat_session(session_id, None, None, Some(10), None, None, None)
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.title.as_deref(), Some("Original title"));
        assert_eq!(s.preview.as_deref(), Some("Original preview"));
        assert_eq!(s.message_count, 10);
    }

    // ====================================================================
    // backfill_chat_session_previews via mock — returns 0 (mock has no events)
    // ====================================================================

    #[tokio::test]
    async fn test_backfill_session_previews_mock_returns_zero() {
        let state = mock_app_state();
        let result = state.neo4j.backfill_chat_session_previews().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    // ====================================================================
    // Title truncation logic (mirrors send_message_internal behavior)
    // ====================================================================

    #[test]
    fn test_title_generation_short_message() {
        let msg = "Hello, help me with my project";
        // < 80 chars → title == message
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title, "Hello, help me with my project");
    }

    #[test]
    fn test_title_generation_long_message() {
        let msg = "a".repeat(100);
        // > 80 chars → truncated to 77 + "..."
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title.chars().count(), 80);
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_preview_generation_short_message() {
        let msg = "Short message";
        let preview = if msg.chars().count() > 200 {
            let truncated: String = msg.chars().take(197).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(preview, "Short message");
    }

    #[test]
    fn test_preview_generation_long_message() {
        let msg = "b".repeat(300);
        let preview = if msg.chars().count() > 200 {
            let truncated: String = msg.chars().take(197).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(preview.chars().count(), 200);
        assert!(preview.ends_with("..."));
    }

    #[test]
    fn test_title_generation_utf8_multibyte() {
        // 90 chars with accented characters
        let msg: String = "é".repeat(90);
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title.chars().count(), 80);
        assert!(title.ends_with("..."));
    }

    // ========================================================================
    // NATS RPC routing tests
    // ========================================================================

    #[tokio::test]
    async fn test_try_remote_send_no_nats() {
        // ChatManager without NATS → try_remote_send should return Ok(false) immediately
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No NATS configured → should return false (fallback needed)
        assert!(manager.nats.is_none());
        let result = manager
            .try_remote_send("any-session-id", "hello", "user_message")
            .await;
        assert!(result.is_ok());
        assert!(!result.unwrap(), "Should return false without NATS");
    }

    #[tokio::test]
    async fn test_try_remote_send_all_message_types_no_nats() {
        // Verify all message types return Ok(false) without NATS
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        for msg_type in &["user_message", "permission_response", "input_response"] {
            let result = manager
                .try_remote_send("session-123", "test", msg_type)
                .await
                .unwrap();
            assert!(!result, "Should return false for {} without NATS", msg_type);
        }
    }

    #[tokio::test]
    async fn test_interrupt_no_local_session_no_nats() {
        // interrupt() should succeed silently when session is not local and no NATS
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No session active, no NATS → should return Ok (no error)
        let result = manager.interrupt("nonexistent-session").await;
        assert!(
            result.is_ok(),
            "interrupt should not error when session is not local"
        );
    }

    #[tokio::test]
    async fn test_spawn_nats_rpc_listener_noop_without_nats() {
        // Verifies that spawn_nats_rpc_listener is a no-op without NATS (no panic, no error)
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // This should be a complete no-op — no NATS means early return
        manager.spawn_nats_rpc_listener("test-session", manager.active_sessions.clone());
        // If we get here without panic, the test passes
    }

    #[tokio::test]
    async fn test_routing_decision_local_active_session() {
        // When a session is active locally, is_session_active returns true
        // → the routing logic should choose send_message (local path)
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No active sessions → is_session_active should return false
        assert!(!manager.is_session_active("some-session").await);

        // try_remote_send without NATS → false
        let remote = manager
            .try_remote_send("some-session", "hello", "user_message")
            .await
            .unwrap();
        assert!(!remote);

        // This confirms the routing: not local + not remote = resume_session fallback
    }
}
