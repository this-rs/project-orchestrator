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
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    PermissionMode,
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
}

/// Manages chat sessions and their lifecycle
pub struct ChatManager {
    pub(crate) graph: Arc<dyn GraphStore>,
    #[allow(dead_code)]
    pub(crate) search: Arc<dyn SearchStore>,
    pub(crate) config: ChatConfig,
    pub(crate) active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
}

impl ChatManager {
    /// Create a new ChatManager
    pub fn new(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
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
            .permission_mode(PermissionMode::AcceptEdits)
            .max_turns(self.config.max_turns)
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

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx_clone,
                message,
                session_id_str.clone(),
                graph,
                active_sessions,
                interrupt_flag,
            )
            .await;
        });

        Ok(CreateSessionResponse {
            session_id: session_id.to_string(),
            stream_url: format!("/api/chat/{}/stream", session_id),
        })
    }

    /// Internal: send a message to the client and stream the response to broadcast
    async fn stream_response(
        client: Arc<Mutex<InteractiveClient>>,
        events_tx: broadcast::Sender<ChatEvent>,
        prompt: String,
        session_id: String,
        graph: Arc<dyn GraphStore>,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        interrupt_flag: Arc<AtomicBool>,
    ) {
        // Send message
        {
            let mut c = client.lock().await;
            if let Err(e) = c.send_message(prompt).await {
                let _ = events_tx.send(ChatEvent::Error {
                    message: format!("Failed to send message: {}", e),
                });
                return;
            }
        }

        // Check interrupt before starting stream
        if interrupt_flag.load(Ordering::SeqCst) {
            debug!(
                "Interrupt flag set before streaming for session {}",
                session_id
            );
            return;
        }

        // Stream response in a block so that the lock (and stream borrow) are dropped at block end
        {
            let mut c = client.lock().await;
            let stream = c.receive_response_stream().await;
            tokio::pin!(stream);

            while let Some(result) = stream.next().await {
                // Check interrupt flag on every iteration — this is the key fix
                if interrupt_flag.load(Ordering::SeqCst) {
                    info!(
                        "Interrupt flag detected in stream loop for session {}",
                        session_id
                    );
                    break;
                }

                match result {
                    Ok(msg) => {
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

                        let events = Self::message_to_events(&msg);
                        for event in events {
                            let _ = events_tx.send(event);
                        }
                    }
                    Err(e) => {
                        error!("Stream error for session {}: {}", session_id, e);
                        let _ = events_tx.send(ChatEvent::Error {
                            message: format!("Stream error: {}", e),
                        });
                        break;
                    }
                }
            }
            // stream and c are dropped here, releasing the Mutex
        }

        // If interrupted, now that the lock is released, send the interrupt signal to the CLI
        if interrupt_flag.load(Ordering::SeqCst) {
            debug!(
                "Stream loop broken by interrupt, sending interrupt signal to CLI for session {}",
                session_id
            );
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
        // Check if session is active
        let (client, events_tx, interrupt_flag) = {
            let sessions = self.active_sessions.read().await;
            let session = sessions
                .get(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            // Reset interrupt flag for new message
            session.interrupt_flag.store(false, Ordering::SeqCst);
            (
                session.client.clone(),
                session.events_tx.clone(),
                session.interrupt_flag.clone(),
            )
        };

        // Update last activity
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(active) = sessions.get_mut(session_id) {
                active.last_activity = Instant::now();
            }
        }

        // Update message count in Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            // Get current count and increment
            if let Ok(Some(node)) = self.graph.get_chat_session(uuid).await {
                let _ = self
                    .graph
                    .update_chat_session(uuid, None, None, Some(node.message_count + 1), None)
                    .await;
            }
        }

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
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
                },
            );
            interrupt_flag
        };

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
            )
            .await;
        });

        Ok(())
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
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        assert_eq!(
            manager.resolve_model(Some("claude-sonnet-4-20250514")),
            "claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_resolve_model_default() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        assert_eq!(manager.resolve_model(None), "claude-opus-4-6");
    }

    #[tokio::test]
    async fn test_build_system_prompt_no_project() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        let prompt = manager.build_system_prompt(None).await;
        assert!(prompt.contains("Project Orchestrator"));
        assert!(!prompt.contains("Projet actif"));
    }

    #[tokio::test]
    async fn test_build_system_prompt_with_project() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let manager = ChatManager::new(state.neo4j, state.meili, test_config());
        let prompt = manager.build_system_prompt(Some(&project.slug)).await;

        assert!(prompt.contains("Projet actif"));
        assert!(prompt.contains(&project.name));
    }

    #[tokio::test]
    async fn test_session_not_active_by_default() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        assert!(!manager.is_session_active("nonexistent").await);
    }

    // ====================================================================
    // build_options
    // ====================================================================

    #[test]
    fn test_build_options_basic() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

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
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

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
        let manager = ChatManager::new(state.neo4j, state.meili, config);

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
    // active_session_count
    // ====================================================================

    #[tokio::test]
    async fn test_active_session_count_empty() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        assert_eq!(manager.active_session_count().await, 0);
    }

    // ====================================================================
    // subscribe / interrupt / close errors for missing sessions
    // ====================================================================

    #[tokio::test]
    async fn test_subscribe_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        let result = manager.subscribe("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_interrupt_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_close_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        let result = manager.close_session("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_message_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

        let result = manager.send_message("nonexistent", "hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resume_session_invalid_uuid() {
        let state = mock_app_state();
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

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
        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

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

        let manager = ChatManager::new(state.neo4j, state.meili, test_config());

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
            .update_chat_session(session.id, None, Some("Title only".into()), None, None)
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
            .update_chat_session(uuid::Uuid::new_v4(), None, None, None, None)
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
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.cli_session_id, session.cli_session_id);
        assert_eq!(deserialized.message_count, 10);
        assert_eq!(deserialized.total_cost_usd, Some(1.50));
    }
}
