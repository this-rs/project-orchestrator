//! ChatManager — orchestrates Claude Code CLI sessions via Nexus SDK
//!
//! Manages active InteractiveClient sessions with auto-resume for inactive ones.
//!
//! Architecture:
//! - Each session spawns an `InteractiveClient` (Nexus SDK) subprocess
//! - Messages are streamed via `broadcast::channel` to SSE subscribers
//! - Inactive sessions are persisted in Neo4j with `cli_session_id` for resume
//! - A cleanup task periodically closes timed-out sessions

use super::config::ChatConfig;
use super::types::{ChatEvent, ChatRequest, CreateSessionResponse};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatSessionNode;
use crate::neo4j::GraphStore;
use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use nexus_claude::{
    memory::{ContextInjector, ConversationMemoryManager, LoadedConversation, MemoryConfig},
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    PermissionMode, StreamDelta, StreamEventData,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Broadcast channel buffer size for SSE subscribers
const BROADCAST_BUFFER: usize = 256;

/// An active chat session with a live Claude CLI subprocess
pub struct ActiveSession {
    /// Broadcast sender for SSE subscribers
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
        }
    }

    /// Resolve the model to use: request > config default
    pub fn resolve_model(&self, request_model: Option<&str>) -> String {
        request_model
            .map(|m| m.to_string())
            .unwrap_or_else(|| self.config.default_model.clone())
    }

    /// Build the system prompt with project context
    pub async fn build_system_prompt(&self, project_slug: Option<&str>) -> String {
        let mut prompt = String::from(
            "Tu es un assistant de développement intégré au Project Orchestrator. \
             Tu as accès à tous les outils MCP pour gérer les plans, tasks, milestones, \
             notes et explorer le code. Utilise-les proactivement quand l'utilisateur \
             te demande de planifier, organiser ou analyser du code.\n\n",
        );

        if let Some(slug) = project_slug {
            // Fetch project context from Neo4j
            if let Ok(Some(project)) = self.graph.get_project_by_slug(slug).await {
                prompt.push_str(&format!("## Projet actif : {} ({})\n", project.name, slug));
                prompt.push_str(&format!("Root: {}\n\n", project.root_path));
            }

            // Fetch active plans
            if let Ok(plans) = self.graph.list_active_plans().await {
                if !plans.is_empty() {
                    prompt.push_str("## Plans en cours\n");
                    for plan in &plans {
                        prompt.push_str(&format!("- {} ({:?})\n", plan.title, plan.status));
                    }
                    prompt.push('\n');
                }
            }
        }

        prompt
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
            Message::Assistant { message } => {
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
                _ => vec![],
            },
            Message::System { subtype, data } => {
                debug!("System message: {} — {:?}", subtype, data);
                vec![]
            }
            Message::User { .. } => vec![],
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
            .build_system_prompt(request.project_slug.as_deref())
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
                .update_chat_session(session_id, None, None, None, None, Some(conversation_id))
                .await;

            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        info!("Created chat session {} with model {}", session_id, model);

        // Create broadcast channel
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);
        let client = Arc::new(Mutex::new(client));

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
                },
            );
            interrupt_flag
        };

        // Send the initial message and start streaming in a background task
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let message = request.message.clone();
        let events_tx_clone = events_tx.clone();
        let injector = self.context_injector.clone();

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
            )
            .await;
        });

        Ok(CreateSessionResponse {
            session_id: session_id.to_string(),
            stream_url: format!("/api/chat/{}/stream", session_id),
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
    ) {
        // Record user message in memory manager
        if let Some(ref mm) = memory_manager {
            let mut mm = mm.lock().await;
            mm.record_user_message(&prompt);
        }
        // Wait for at least one SSE subscriber before sending the message to the CLI.
        // Without this, events emitted before the frontend connects are lost (broadcast
        // has no replay buffer). Poll every 50ms, timeout after 10s.
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while events_tx.receiver_count() == 0 {
            if Instant::now() >= deadline {
                warn!(
                    "No SSE subscriber connected within 10s for session {}, proceeding anyway",
                    session_id
                );
                break;
            }
            if interrupt_flag.load(Ordering::SeqCst) {
                debug!(
                    "Interrupt flag set while waiting for subscriber for session {}",
                    session_id
                );
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Use send_and_receive_stream() for real-time token streaming.
        // The returned stream borrows &mut InteractiveClient, so we must hold
        // the Mutex lock for the entire stream duration. This is safe because:
        // - Each message spawns stream_response in its own tokio::spawn
        // - send_message() creates a new broadcast channel for each follow-up
        // - The lock is only held during active streaming (not between messages)
        // Collect assistant text for memory recording
        let mut assistant_text_parts: Vec<String> = Vec::new();

        {
            let mut c = client.lock().await;
            let stream_result = c.send_and_receive_stream(prompt).await;

            let mut stream = match stream_result {
                Ok(s) => std::pin::pin!(s),
                Err(e) => {
                    error!("Error starting stream for session {}: {}", session_id, e);
                    let _ = events_tx.send(ChatEvent::Error {
                        message: format!("Error: {}", e),
                    });
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
                        if let Message::StreamEvent {
                            event:
                                StreamEventData::ContentBlockDelta {
                                    delta: StreamDelta::TextDelta { ref text },
                                    ..
                                },
                            ..
                        } = msg
                        {
                            let _ = events_tx.send(ChatEvent::StreamDelta { text: text.clone() });
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
                            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                                let _ = graph
                                    .update_chat_session(
                                        uuid,
                                        Some(cli_sid.clone()),
                                        None,
                                        Some(1),
                                        *cost,
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
                        if let Message::Assistant { message: ref am } = msg {
                            for block in &am.content {
                                if let ContentBlock::Text(t) = block {
                                    assistant_text_parts.push(t.text.clone());
                                }
                            }
                        }

                        // For all other messages, use existing converter
                        let events = Self::message_to_events(msg);
                        for event in events {
                            let _ = events_tx.send(event);
                        }
                    }
                    Err(e) => {
                        error!("Stream error for session {}: {}", session_id, e);
                        let _ = events_tx.send(ChatEvent::Error {
                            message: format!("Error: {}", e),
                        });
                        break;
                    }
                }
            }
        } // Lock released here after stream completes

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
                            warn!(
                                "Failed to store messages for session {}: {}",
                                session_id, e
                            );
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

        debug!("Stream completed for session {}", session_id);
    }

    /// Send a follow-up message to an existing session
    pub async fn send_message(&self, session_id: &str, message: &str) -> Result<()> {
        // Create a NEW broadcast channel for this message so that the fresh /stream
        // subscriber receives all events (old subscribers just stop getting new data).
        let (client, events_tx, interrupt_flag, memory_manager) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

            // Reset interrupt flag for new message
            session.interrupt_flag.store(false, Ordering::SeqCst);

            // Replace the broadcast channel so new /stream subscribers get fresh events
            let (new_tx, _) = broadcast::channel(BROADCAST_BUFFER);
            session.events_tx = new_tx.clone();
            session.last_activity = Instant::now();

            (
                session.client.clone(),
                new_tx,
                session.interrupt_flag.clone(),
                session.memory_manager.clone(),
            )
        };

        // Update message count in Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            // Get current count and increment
            if let Ok(Some(node)) = self.graph.get_chat_session(uuid).await {
                let _ = self
                    .graph
                    .update_chat_session(uuid, None, None, Some(node.message_count + 1), None, None)
                    .await;
            }
        }

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();

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
            )
            .await;
        });

        Ok(())
    }

    /// Resume a previously inactive session by creating a new InteractiveClient with --resume
    pub async fn resume_session(&self, session_id: &str, message: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Load session from Neo4j
        let session_node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found in database", session_id))?;

        let cli_session_id = session_node
            .cli_session_id
            .as_deref()
            .ok_or_else(|| anyhow!("Session {} has no CLI session ID for resume", session_id))?;

        info!(
            "Resuming session {} with CLI ID {}",
            session_id, cli_session_id
        );

        // Build options with resume flag
        let system_prompt = self
            .build_system_prompt(session_node.project_slug.as_deref())
            .await;
        let options = self.build_options(
            &session_node.cwd,
            &session_node.model,
            &system_prompt,
            Some(cli_session_id),
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
                    )
                    .await;
                mm
            };
            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        // Register as active
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            let interrupt_flag = Arc::new(AtomicBool::new(false));
            sessions.insert(
                session_id.to_string(),
                ActiveSession {
                    events_tx: events_tx.clone(),
                    last_activity: Instant::now(),
                    cli_session_id: Some(cli_session_id.to_string()),
                    client: client.clone(),
                    interrupt_flag: interrupt_flag.clone(),
                    memory_manager: memory_manager.clone(),
                },
            );
            interrupt_flag
        };

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();

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
            )
            .await;
        });

        Ok(())
    }

    /// Retrieve message history for a session via ContextInjector
    pub async fn get_session_messages(
        &self,
        session_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<LoadedConversation> {
        let injector = self
            .context_injector
            .as_ref()
            .ok_or_else(|| anyhow!("Message history not available (ContextInjector not initialized)"))?;

        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Fetch conversation_id from Neo4j
        let session_node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        let conversation_id = session_node
            .conversation_id
            .ok_or_else(|| anyhow!("Session {} has no conversation_id", session_id))?;

        injector
            .load_conversation(&conversation_id, limit, offset)
            .await
            .context("Failed to load conversation messages")
    }

    /// Subscribe to a session's events (for SSE streaming)
    pub async fn subscribe(&self, session_id: &str) -> Result<broadcast::Receiver<ChatEvent>> {
        let sessions = self.active_sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        Ok(session.events_tx.subscribe())
    }

    /// Interrupt the current operation in a session.
    ///
    /// Sets the interrupt flag, which causes the stream loop to break and release the
    /// client lock. The stream loop then sends the actual interrupt signal to the CLI.
    /// This is instantaneous — no waiting for the Mutex.
    pub async fn interrupt(&self, session_id: &str) -> Result<()> {
        let interrupt_flag = {
            let sessions = self.active_sessions.read().await;
            let session = sessions
                .get(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.interrupt_flag.clone()
        };

        // Set the flag — the stream loop checks this on every iteration
        // and will break + release the lock + send the actual interrupt signal
        interrupt_flag.store(true, Ordering::SeqCst);

        info!("Interrupt flag set for session {}", session_id);
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
            max_turns: 10,
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

        let prompt = manager.build_system_prompt(None).await;
        assert!(prompt.contains("Project Orchestrator"));
        assert!(!prompt.contains("Projet actif"));
    }

    #[tokio::test]
    async fn test_build_system_prompt_with_project() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let prompt = manager.build_system_prompt(Some(&project.slug)).await;

        assert!(prompt.contains("Projet actif"));
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
            },
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
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
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::StreamDelta { text } if text == "Hello"));
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
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_stop() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageStop,
            session_id: None,
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
        let prompt = manager.build_system_prompt(Some(&project.slug)).await;

        assert!(prompt.contains("Projet actif"));
        assert!(prompt.contains(&project.name));
        assert!(prompt.contains("Plans en cours"));
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
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent").await;
        assert!(result.is_err());
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
    async fn test_resume_session_no_cli_session_id() {
        let state = mock_app_state();
        let session = test_chat_session(None); // no cli_session_id
        state.neo4j.create_chat_session(&session).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .resume_session(&session.id.to_string(), "hello")
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no CLI session ID"));
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
            .update_chat_session(session.id, None, Some("Title only".into()), None, None, None)
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
            .update_chat_session(uuid::Uuid::new_v4(), None, None, None, None, None)
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
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.cli_session_id, session.cli_session_id);
        assert_eq!(deserialized.message_count, 10);
        assert_eq!(deserialized.total_cost_usd, Some(1.50));
    }
}
