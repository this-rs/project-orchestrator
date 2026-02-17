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
use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use nexus_claude::{
    build_hook_response_json, dispatch_hook_from_registry, is_hook_callback,
    memory::{ContextInjector, ConversationMemoryManager, MemoryConfig},
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    PermissionMode, StreamDelta, StreamEventData,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
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
    /// Current permission mode for this session (updated on mid-session changes)
    pub permission_mode: Option<String>,
    /// Current model for this session (updated on mid-session model changes)
    pub model: Option<String>,
    /// SDK control receiver for permission requests (`can_use_tool`).
    /// Taken once from `InteractiveClient::take_sdk_control_receiver()` at session
    /// creation and reused across all `stream_response` invocations. Wrapped in
    /// `Arc<Mutex<Option<...>>>` so each `stream_response` can temporarily take
    /// ownership during streaming, then put it back when the stream ends.
    pub sdk_control_rx:
        Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<serde_json::Value>>>>,
    /// Cloned stdin sender for writing control responses (e.g., permission
    /// allow/deny) to the CLI subprocess **without** taking the `client` lock.
    /// This is critical because `stream_response` holds the client lock for the
    /// entire duration of streaming — if `send_permission_response` tried to
    /// take the same lock, it would deadlock.
    pub stdin_tx: Option<tokio::sync::mpsc::Sender<String>>,
    /// Cancellation token for NATS listener tasks spawned by this session.
    /// When the session is replaced (e.g., by `resume_session`), the old token
    /// is cancelled so that stale NATS listeners (interrupt, snapshot, RPC)
    /// shut down instead of accumulating across resumes/restarts.
    pub nats_cancel: CancellationToken,
    /// Stores the original tool input for pending permission requests.
    /// Key: request_id, Value: the tool input JSON.
    /// When the user responds Allow, we include this input in `updatedInput`
    /// so the CLI doesn't lose the original command/parameters.
    pub pending_permission_inputs:
        Arc<tokio::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
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
    /// Runtime-mutable permission config (updated via REST API)
    pub(crate) permission_config: Arc<RwLock<super::config::PermissionConfig>>,
    /// Path to config.yaml for persisting permission changes (None = no persistence)
    pub(crate) config_yaml_path: Option<std::path::PathBuf>,
}

// ============================================================================
// CompactionNotifier — HookCallback that emits ChatEvent::CompactionStarted
// ============================================================================

/// Hook callback that emits a `ChatEvent::CompactionStarted` when the CLI
/// triggers a `PreCompact` event (context window compaction is about to start).
///
/// This gives the frontend a real-time notification so it can display a spinner
/// instead of leaving the user staring at silence during compaction.
///
/// The callback always returns `continue_: true` — it observes, never blocks.
pub(crate) struct CompactionNotifier {
    /// Broadcast sender for chat events (same channel as stream_response uses)
    events_tx: broadcast::Sender<ChatEvent>,
}

impl CompactionNotifier {
    pub fn new(events_tx: broadcast::Sender<ChatEvent>) -> Self {
        Self { events_tx }
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for CompactionNotifier {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        if let nexus_claude::HookInput::PreCompact(pre_compact) = input {
            let event = ChatEvent::CompactionStarted {
                trigger: pre_compact.trigger.clone(),
            };
            // Best-effort broadcast — receivers may have been dropped (no subscribers)
            let _ = self.events_tx.send(event);
            info!(
                trigger = %pre_compact.trigger,
                "PreCompact hook fired — emitted CompactionStarted event"
            );
        }
        // Always allow compaction to proceed
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                ..Default::default()
            },
        ))
    }
}

impl ChatManager {
    /// Create a ChatManager without memory support (for tests or when Meilisearch is unavailable)
    pub fn new_without_memory(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        let permission_config = Arc::new(RwLock::new(config.permission.clone()));
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            context_injector: None,
            memory_config: None,
            event_emitter: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
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

        let permission_config = Arc::new(RwLock::new(config.permission.clone()));
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            context_injector,
            memory_config: Some(memory_config),
            event_emitter: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
        }
    }

    /// Set the event emitter for CRUD events (streaming status notifications)
    pub fn with_event_emitter(mut self, emitter: Arc<dyn crate::events::EventEmitter>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Set the config.yaml path for persisting permission config changes.
    pub fn with_config_yaml_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_yaml_path = Some(path);
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

    // ========================================================================
    // Runtime permission config (GET / UPDATE via REST API)
    // ========================================================================

    /// Get a clone of the current runtime permission config.
    pub async fn get_permission_config(&self) -> super::config::PermissionConfig {
        self.permission_config.read().await.clone()
    }

    /// Update the runtime permission config.
    ///
    /// Validates the mode string before applying. Returns an error if the mode
    /// is not one of the valid values ("default", "acceptEdits", "plan", "bypassPermissions").
    /// New sessions will pick up the updated config immediately.
    /// Active sessions keep their original config (no mid-session changes).
    ///
    /// When a `config_yaml_path` is set, the updated config is also persisted
    /// to disk atomically (write to .tmp then rename).
    pub async fn update_permission_config(
        &self,
        new_config: super::config::PermissionConfig,
    ) -> Result<super::config::PermissionConfig> {
        if !super::config::PermissionConfig::is_valid_mode(&new_config.mode) {
            return Err(anyhow!(
                "Invalid permission mode '{}'. Valid modes: {:?}",
                new_config.mode,
                super::config::PermissionConfig::valid_modes()
            ));
        }
        let mut perm = self.permission_config.write().await;
        *perm = new_config;
        let result = perm.clone();
        // Drop the lock before doing I/O
        drop(perm);

        // Persist to config.yaml if a path is configured
        if let Some(ref yaml_path) = self.config_yaml_path {
            if let Err(e) = Self::persist_permission_to_yaml(yaml_path, &result) {
                // Log but don't fail the API call — in-memory update succeeded
                error!(
                    path = %yaml_path.display(),
                    error = %e,
                    "Failed to persist permission config to config.yaml"
                );
            } else {
                info!(
                    path = %yaml_path.display(),
                    mode = %result.mode,
                    "Permission config persisted to config.yaml"
                );
            }
        }

        Ok(result)
    }

    /// Persist permission config to config.yaml using surgical YAML modification.
    ///
    /// Reads the existing file as a `serde_yaml::Value` tree, updates only the
    /// `chat.permissions` subtree, and writes back atomically (tmp + rename).
    /// This preserves all other config sections (auth, server, neo4j, etc.)
    /// without needing `Serialize` on those structs.
    fn persist_permission_to_yaml(
        yaml_path: &std::path::Path,
        permission: &super::config::PermissionConfig,
    ) -> Result<()> {
        use std::io::Write;

        // 1. Read existing YAML as a Value tree (or start from empty mapping)
        let mut doc: serde_yaml::Value = if yaml_path.exists() {
            let contents = std::fs::read_to_string(yaml_path)
                .with_context(|| format!("Reading {}", yaml_path.display()))?;
            serde_yaml::from_str(&contents)
                .with_context(|| format!("Parsing {}", yaml_path.display()))?
        } else {
            serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
        };

        // 2. Ensure doc is a mapping
        let root = doc
            .as_mapping_mut()
            .ok_or_else(|| anyhow!("config.yaml root is not a YAML mapping"))?;

        // 3. Ensure chat section exists as a mapping
        let chat_key = serde_yaml::Value::String("chat".into());
        if !root.contains_key(&chat_key) {
            root.insert(
                chat_key.clone(),
                serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
            );
        }
        let chat_section = root
            .get_mut(&chat_key)
            .and_then(|v| v.as_mapping_mut())
            .ok_or_else(|| anyhow!("chat section is not a YAML mapping"))?;

        // 4. Serialize PermissionConfig to a YAML Value and insert
        let perm_value = serde_yaml::to_value(permission)
            .context("Serializing PermissionConfig to YAML value")?;
        chat_section.insert(serde_yaml::Value::String("permissions".into()), perm_value);

        // 5. Serialize the full document back to YAML string
        let yaml_str =
            serde_yaml::to_string(&doc).context("Serializing config document to YAML")?;

        // 6. Atomic write: write to .tmp then rename
        let tmp_path = yaml_path.with_extension("yaml.tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .with_context(|| format!("Creating {}", tmp_path.display()))?;
            file.write_all(yaml_str.as_bytes())
                .with_context(|| format!("Writing {}", tmp_path.display()))?;
            file.sync_all()
                .with_context(|| format!("Syncing {}", tmp_path.display()))?;
        }
        std::fs::rename(&tmp_path, yaml_path).with_context(|| {
            format!("Renaming {} → {}", tmp_path.display(), yaml_path.display())
        })?;

        Ok(())
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
        cancel: CancellationToken,
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

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS interrupt listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(_msg) = msg else { break; };

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
                }
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
        cancel: CancellationToken,
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

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS snapshot responder cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(msg) = msg else { break; };

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
        cancel: CancellationToken,
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

            loop {
                let msg = tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS RPC listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        match msg {
                            Some(m) => m,
                            None => break,
                        }
                    }
                };
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
                                session.sdk_control_rx.clone(),
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
                        sdk_control_rx,
                    )) => {
                        let message = &request.message;

                        // Route based on message_type
                        if request.message_type == "control_response" {
                            // Permission control response — send directly to CLI subprocess
                            // via SDK control protocol. Do NOT persist or broadcast.
                            let allow: bool = serde_json::from_str::<serde_json::Value>(message)
                                .ok()
                                .and_then(|v| v.get("allow").and_then(|a| a.as_bool()))
                                .unwrap_or(false);

                            info!(
                                session_id = %session_id,
                                allow,
                                "NATS RPC: Sending permission control response to CLI"
                            );

                            let response_json = serde_json::json!({ "allow": allow });
                            let mut cli = client.lock().await;
                            match cli.send_control_response(response_json).await {
                                Ok(()) => crate::events::ChatRpcResponse {
                                    success: true,
                                    error: None,
                                },
                                Err(e) => crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(format!("Failed to send control response: {}", e)),
                                },
                            }
                        } else if is_streaming.load(Ordering::SeqCst) {
                            // If streaming → queue the message (will be drained by stream_response)
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
                                    data: serde_json::to_string(&ChatEvent::UserMessage {
                                        content: message.to_string(),
                                    })
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
                                    sdk_control_rx,
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
                context_to_markdown(&ctx, Some(user_message))
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

        // Build options: no MCP server, max_turns=1, just text generation.
        // NOTE: Intentionally hardcoded to BypassPermissions — this is an internal
        // one-shot context refinement call with no user-facing tool interaction.
        // It must NOT use the user-configured permission mode.
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

    /// Build `ClaudeCodeOptions` for a new or resumed session.
    ///
    /// `permission_mode_override`: if Some, overrides the global config permission mode
    /// for this specific session (e.g. user chose a different mode for this session).
    #[allow(deprecated)]
    pub async fn build_options(
        &self,
        cwd: &str,
        model: &str,
        system_prompt: &str,
        resume_id: Option<&str>,
        permission_mode_override: Option<&str>,
        hooks: Option<std::collections::HashMap<String, Vec<nexus_claude::HookMatcher>>>,
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

        // Read the runtime-mutable permission config (updated via REST API)
        let perm_config = self.permission_config.read().await;

        // Use per-session override if provided, otherwise use global config
        let effective_permission = match permission_mode_override {
            Some(mode_str) => {
                let override_config = super::config::PermissionConfig {
                    mode: mode_str.to_string(),
                    ..Default::default()
                };
                override_config.to_nexus_mode()
            }
            None => perm_config.to_nexus_mode(),
        };

        let mut builder = ClaudeCodeOptions::builder()
            .model(model)
            .cwd(cwd)
            .system_prompt(system_prompt)
            .permission_mode(effective_permission)
            .max_turns(self.config.max_turns)
            .include_partial_messages(true)
            .permission_prompt_tool_name("stdio")
            .add_mcp_server("project-orchestrator", mcp_config);

        // Wire allowed/disallowed tool patterns from config
        if !perm_config.allowed_tools.is_empty() {
            builder = builder.allowed_tools(perm_config.allowed_tools.clone());
        }
        if !perm_config.disallowed_tools.is_empty() {
            builder = builder.disallowed_tools(perm_config.disallowed_tools.clone());
        }

        if let Some(id) = resume_id {
            builder = builder.resume(id);
        }

        if let Some(hook_map) = hooks {
            builder = builder.hooks(hook_map);
        }

        builder.build()
    }

    // ========================================================================
    // Message → ChatEvent conversion
    // ========================================================================

    /// Convert a Nexus SDK `Message` to a list of `ChatEvent`s
    pub fn message_to_events(msg: &Message) -> Vec<ChatEvent> {
        // Extract parent_tool_use_id from the Message (sidechain indicator).
        // When present, this event originated from a sub-agent spawned by a Task tool.
        let parent = msg.parent_tool_use_id().map(|s| s.to_string());

        match msg {
            Message::Assistant { message, .. } => {
                let mut events = Vec::new();
                for block in &message.content {
                    match block {
                        ContentBlock::Text(t) => {
                            events.push(ChatEvent::AssistantText {
                                content: t.text.clone(),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                        ContentBlock::Thinking(t) => {
                            events.push(ChatEvent::Thinking {
                                content: t.thinking.clone(),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                        ContentBlock::ToolUse(t) => {
                            events.push(ChatEvent::ToolUse {
                                id: t.id.clone(),
                                tool: t.name.clone(),
                                input: t.input.clone(),
                                parent_tool_use_id: parent.clone(),
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
                                parent_tool_use_id: parent.clone(),
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
                subtype,
                is_error,
                num_turns,
                result,
                ..
            } => {
                vec![ChatEvent::Result {
                    session_id: session_id.clone(),
                    duration_ms: *duration_ms as u64,
                    cost_usd: *total_cost_usd,
                    subtype: subtype.clone(),
                    is_error: *is_error,
                    num_turns: Some(*num_turns),
                    result_text: result.clone(),
                }]
            }
            Message::StreamEvent { event, .. } => match event {
                StreamEventData::ContentBlockDelta {
                    delta: StreamDelta::TextDelta { text },
                    ..
                } => {
                    vec![ChatEvent::StreamDelta {
                        text: text.clone(),
                        parent_tool_use_id: parent,
                    }]
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
                            parent_tool_use_id: parent,
                        }]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            },
            Message::System { subtype, data } => {
                match subtype.as_str() {
                    "init" => {
                        // Extract session metadata from init system message
                        let cli_session_id = data
                            .get("session_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let model = data
                            .get("model")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let tools = data
                            .get("tools")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let mcp_servers = data
                            .get("mcp_servers")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let permission_mode = data
                            .get("permissionMode")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        vec![ChatEvent::SystemInit {
                            cli_session_id,
                            model,
                            tools,
                            mcp_servers,
                            permission_mode,
                        }]
                    }
                    "compact_boundary" => {
                        // Extract compact metadata from data.compact_metadata
                        let metadata = data.get("compact_metadata");
                        let trigger = metadata
                            .and_then(|m| m.get("trigger"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("auto")
                            .to_string();
                        let pre_tokens = metadata
                            .and_then(|m| m.get("pre_tokens"))
                            .and_then(|v| v.as_u64());
                        vec![ChatEvent::CompactBoundary {
                            trigger,
                            pre_tokens,
                        }]
                    }
                    _ => {
                        debug!("Unhandled system message: {} — {:?}", subtype, data);
                        vec![]
                    }
                }
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
                                parent_tool_use_id: parent.clone(),
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
            permission_mode: request.permission_mode.clone(),
        };
        self.graph
            .create_chat_session(&session_node)
            .await
            .context("Failed to persist chat session")?;

        // Create broadcast channel early so CompactionNotifier can use the sender
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);

        // Build PreCompact hook → CompactionNotifier broadcasts ChatEvent::CompactionStarted
        let compaction_hooks = {
            let notifier = CompactionNotifier::new(events_tx.clone());
            let mut hooks = std::collections::HashMap::new();
            hooks.insert(
                "PreCompact".to_string(),
                vec![nexus_claude::HookMatcher {
                    matcher: None,
                    hooks: vec![std::sync::Arc::new(notifier)],
                }],
            );
            hooks
        };

        // Build options and create InteractiveClient
        let options = self
            .build_options(
                &request.cwd,
                &model,
                &system_prompt,
                None,
                request.permission_mode.as_deref(),
                Some(compaction_hooks),
            )
            .await;
        let mut client = InteractiveClient::new(options)
            .map_err(|e| anyhow!("Failed to create InteractiveClient: {}", e))?;

        client
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect InteractiveClient: {}", e))?;

        // Initialize hooks with the CLI (sends PreCompact, etc. registrations).
        // Must be called AFTER connect() and BEFORE take_sdk_control_receiver().
        // Graceful: warn on failure but don't abort the session.
        if let Err(e) = client.initialize_hooks().await {
            warn!(
                session_id = %session_id,
                "Failed to initialize hooks with CLI (non-fatal): {}",
                e
            );
        }

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

        // Take the SDK control receiver ONCE at session creation.
        // This channel receives `can_use_tool` permission requests from the CLI subprocess.
        // It must be taken before wrapping the client in Arc<Mutex<>> so it can be
        // reused across all stream_response invocations for this session.
        let sdk_control_rx = client.take_sdk_control_receiver().await;
        let sdk_control_rx = Arc::new(tokio::sync::Mutex::new(sdk_control_rx));

        // Clone the stdin sender BEFORE wrapping client in Arc<Mutex<>>.
        // This allows send_permission_response to write control responses
        // directly to the CLI subprocess without taking the client lock
        // (which is held by stream_response during streaming → deadlock).
        let stdin_tx = client.clone_stdin_sender().await;

        info!(
            session_id = %session_id,
            has_stdin_tx = stdin_tx.is_some(),
            "Created chat session with model {}",
            model
        );

        let client = Arc::new(Mutex::new(client));

        // Initialize next_seq (new session = start at 1)
        let next_seq = Arc::new(AtomicI64::new(1));
        let pending_messages = Arc::new(Mutex::new(VecDeque::new()));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let streaming_text = Arc::new(Mutex::new(String::new()));
        let streaming_events = Arc::new(Mutex::new(Vec::new()));

        // Register active session — cancel old NATS listeners if session key already exists
        let nats_cancel = CancellationToken::new();
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // Cancel stale NATS listeners from a previous session with the same ID
            if let Some(old_session) = sessions.get(&session_id.to_string()) {
                info!(
                    session_id = %session_id,
                    "Cancelling stale NATS listeners for existing session (create_session replacing)"
                );
                old_session.nats_cancel.cancel();
            }
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
                    permission_mode: request.permission_mode.clone(),
                    model: Some(model.clone()),
                    sdk_control_rx: sdk_control_rx.clone(),
                    stdin_tx,
                    nats_cancel: nats_cancel.clone(),
                    pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                        std::collections::HashMap::new(),
                    )),
                },
            );
            interrupt_flag
        };

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            &session_id.to_string(),
            interrupt_flag.clone(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(
            &session_id.to_string(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(
            &session_id.to_string(),
            self.active_sessions.clone(),
            nats_cancel,
        );

        // Persist the initial user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: request.message.clone(),
            })
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
                sdk_control_rx,
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
        shared_sdk_control_rx: Arc<
            tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<serde_json::Value>>>,
        >,
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

        // Temporarily take the SDK control receiver from the shared slot.
        // This allows us to listen for control protocol messages (e.g., `can_use_tool`
        // permission requests) in parallel with the message stream.
        // The receiver is put back into the shared slot when this stream ends,
        // so the next stream_response invocation can reuse it.
        let mut sdk_control_rx = shared_sdk_control_rx.lock().await.take();

        // Get a handle to the session's pending permission inputs map so we can
        // store the original tool input when a permission request arrives.
        // send_permission_response() reads from this same Arc to retrieve the input.
        //
        // Also clone the stdin_tx sender for auto-allowing AskUserQuestion control
        // requests inline (without going through send_permission_response).
        let (pending_perm_inputs, stdin_tx_for_auto_allow) = {
            let guard = active_sessions.read().await;
            match guard.get(&session_id) {
                Some(s) => (s.pending_permission_inputs.clone(), s.stdin_tx.clone()),
                None => (
                    Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
                    None,
                ),
            }
        };

        // Extract hook callbacks registry BEFORE taking the long-lived client lock.
        // This Arc<RwLock<>> allows dispatching hook_callbacks inside the select! loop
        // without needing to re-lock the client (which is held by the stream).
        let hook_callbacks_registry = {
            let c = client.lock().await;
            c.hook_callbacks()
        };

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
                            parent_tool_use_id: None,
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

            // Track the current parent_tool_use_id from stream events.
            // When a sub-agent is active, its stream Messages carry parent_tool_use_id.
            // We capture this so permission requests (which arrive via a separate control
            // channel without parent info) can be attributed to the correct agent.
            let mut current_parent_tool_use_id: Option<String> = None;

            // Helper closure: process an SDK control message (permission or AskUserQuestion).
            // Returns Some(ChatEvent) if a can_use_tool was parsed, None otherwise.
            // The caller is responsible for adding the event to streaming_events (async)
            // and for auto-allowing AskUserQuestion events via stdin_tx.
            let handle_control_msg = |control_msg: serde_json::Value,
                                      events_to_persist: &mut Vec<ChatEventRecord>,
                                      next_seq: &std::sync::atomic::AtomicI64,
                                      current_parent: Option<String>|
             -> Option<ChatEvent> {
                let event = parse_permission_control_msg(&control_msg, current_parent)?;

                match &event {
                    ChatEvent::PermissionRequest { id, tool, .. } => {
                        info!(
                            session_id = %session_id,
                            tool = %tool,
                            request_id = %id,
                            "Permission request from CLI"
                        );
                    }
                    ChatEvent::AskUserQuestion { id, .. } => {
                        info!(
                            session_id = %session_id,
                            request_id = %id,
                            "AskUserQuestion from CLI (will auto-allow)"
                        );
                    }
                    _ => {}
                }

                // Persist the event
                if let Some(uuid) = session_uuid {
                    let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                    events_to_persist.push(ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq,
                        event_type: event.event_type().to_string(),
                        data: serde_json::to_string(&event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    });
                }

                // Broadcast to WebSocket clients
                emit_chat(event.clone(), &events_tx, &nats, &session_id);
                Some(event)
            };

            // Main stream loop — uses tokio::select! to listen for BOTH stream
            // events AND SDK control messages (permission requests) concurrently.
            //
            // BUG FIX: Previously used `stream.next().await` followed by
            // `rx.try_recv()`. When the CLI blocks waiting for permission approval,
            // it stops sending stream events, so `stream.next()` would never yield
            // and `try_recv()` was never reached — permission requests were lost.
            //
            // Now we select! between the two sources so control messages are
            // processed even when the stream is idle (waiting for permission).
            loop {
                // Check interrupt flag at each iteration
                if interrupt_flag.load(Ordering::SeqCst) {
                    info!(
                        "Interrupt flag detected during stream for session {}",
                        session_id
                    );
                    break;
                }

                let result = if let Some(ref mut rx) = sdk_control_rx {
                    tokio::select! {
                        biased;  // Prioritize control messages over stream events

                        control_msg = rx.recv() => {
                            match control_msg {
                                Some(msg) => {
                                    // Hook callbacks: dispatch via lock-free registry, send response via stdin_tx
                                    if is_hook_callback(&msg) {
                                        let request_id = msg.get("request_id")
                                            .or_else(|| msg.get("request").and_then(|r| r.get("request_id")))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        if let Some(result) = dispatch_hook_from_registry(&msg, &hook_callbacks_registry).await {
                                            info!(session_id = %session_id, request_id = %request_id, "Hook callback dispatched");
                                            if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                let response_json = build_hook_response_json(&request_id, &result);
                                                let _ = tx.send(response_json).await;
                                                debug!(request_id = %request_id, "Hook response sent to CLI");
                                            }
                                        }
                                        continue;
                                    }

                                    if let Some(evt) = handle_control_msg(msg, &mut events_to_persist, &next_seq, current_parent_tool_use_id.clone()) {
                                        // AskUserQuestion: auto-allow the control request so the CLI
                                        // waits for the tool_result (user's answer) instead of blocking.
                                        // Do NOT store in pending_perm_inputs (it's not a permission).
                                        if let ChatEvent::AskUserQuestion { ref id, ref input, .. } = evt {
                                            if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                let control_response = serde_json::json!({
                                                    "type": "control_response",
                                                    "response": {
                                                        "subtype": "success",
                                                        "request_id": id,
                                                        "response": {
                                                            "behavior": "allow",
                                                            "updatedInput": input
                                                        }
                                                    }
                                                });
                                                if let Ok(json) = serde_json::to_string(&control_response) {
                                                    let _ = tx.send(json).await;
                                                    info!(request_id = %id, "Auto-allowed AskUserQuestion control request");
                                                }
                                            }
                                        } else if let ChatEvent::PermissionRequest { ref id, ref input, .. } = evt {
                                            // Regular permission: store original input for later response
                                            if !id.is_empty() {
                                                store_pending_perm_input(&pending_perm_inputs, id, input).await;
                                            }
                                        }
                                        streaming_events.lock().await.push(evt);
                                    }
                                    continue; // Go back to select! for next event
                                }
                                None => {
                                    debug!("SDK control channel closed");
                                    sdk_control_rx = None;
                                    continue;
                                }
                            }
                        }

                        stream_item = stream.next() => {
                            match stream_item {
                                Some(result) => {
                                    // Also drain any buffered control messages
                                    while let Ok(msg) = rx.try_recv() {
                                        // Hook callbacks: dispatch inline (same as select! branch)
                                        if is_hook_callback(&msg) {
                                            let request_id = msg.get("request_id")
                                                .or_else(|| msg.get("request").and_then(|r| r.get("request_id")))
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("")
                                                .to_string();
                                            if let Some(result) = dispatch_hook_from_registry(&msg, &hook_callbacks_registry).await {
                                                info!(session_id = %session_id, request_id = %request_id, "Hook callback dispatched (drain)");
                                                if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                    let response_json = build_hook_response_json(&request_id, &result);
                                                    let _ = tx.try_send(response_json);
                                                }
                                            }
                                            continue;
                                        }

                                        if let Some(evt) = handle_control_msg(msg, &mut events_to_persist, &next_seq, current_parent_tool_use_id.clone()) {
                                            // AskUserQuestion: auto-allow (same logic as above)
                                            if let ChatEvent::AskUserQuestion { ref id, ref input, .. } = evt {
                                                if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                    let control_response = serde_json::json!({
                                                        "type": "control_response",
                                                        "response": {
                                                            "subtype": "success",
                                                            "request_id": id,
                                                            "response": {
                                                                "behavior": "allow",
                                                                "updatedInput": input
                                                            }
                                                        }
                                                    });
                                                    if let Ok(json) = serde_json::to_string(&control_response) {
                                                        // try_send here because we're in a sync-ish drain loop
                                                        let _ = tx.try_send(json);
                                                        info!(request_id = %id, "Auto-allowed AskUserQuestion control request (drain)");
                                                    }
                                                }
                                            } else if let ChatEvent::PermissionRequest { ref id, ref input, .. } = evt {
                                                if !id.is_empty() {
                                                    store_pending_perm_input(&pending_perm_inputs, id, input).await;
                                                }
                                            }
                                            streaming_events.lock().await.push(evt);
                                        }
                                    }
                                    result
                                }
                                None => break, // Stream ended
                            }
                        }
                    }
                } else {
                    // No control channel — just read from stream
                    match stream.next().await {
                        Some(result) => result,
                        None => break,
                    }
                };

                match result {
                    Ok(ref msg) => {
                        // Track the current parent_tool_use_id from every stream message.
                        // This is used by handle_control_msg to attribute permission
                        // requests to the correct sub-agent. We update on every message
                        // (including top-level ones where parent is None) so the tracker
                        // resets correctly when switching between agents.
                        current_parent_tool_use_id =
                            msg.parent_tool_use_id().map(|s| s.to_string());

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
                            let parent = msg.parent_tool_use_id().map(|s| s.to_string());
                            // Accumulate for mid-stream join snapshot
                            streaming_text.lock().await.push_str(text);
                            emit_chat(
                                ChatEvent::StreamDelta {
                                    text: text.clone(),
                                    parent_tool_use_id: parent,
                                },
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

                        // Extract cli_session_id and model from System init message
                        if let Message::System { subtype, ref data } = msg {
                            if subtype == "init" {
                                let cli_sid = data
                                    .get("session_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if let Some(uuid) = session_uuid {
                                    let _ = graph
                                        .update_chat_session(
                                            uuid,
                                            Some(cli_sid.clone()),
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                        )
                                        .await;
                                }
                                // Update active session
                                let mut sessions = active_sessions.write().await;
                                if let Some(active) = sessions.get_mut(&session_id) {
                                    active.cli_session_id = Some(cli_sid);
                                    active.last_activity = Instant::now();
                                }
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
                                ref id,
                                ref input,
                                ref parent_tool_use_id,
                                ..
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
                                                parent_tool_use_id: parent_tool_use_id.clone(),
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
                                        // Note: streaming_text is a flat buffer that doesn't track
                                        // per-agent text. The parent_tool_use_id is set to None here.
                                        // The frontend uses individual streaming_events (which carry
                                        // parent_tool_use_id) for agent grouping, not partial_text.
                                        streaming_events.lock().await.push(
                                            ChatEvent::AssistantText {
                                                content: st.clone(),
                                                parent_tool_use_id: None,
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
                                parent_tool_use_id: None,
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

        // Put the SDK control receiver back into the shared slot so the next
        // stream_response invocation can reuse it (fixes permission requests
        // being silently lost after the first message in a session).
        if sdk_control_rx.is_some() {
            *shared_sdk_control_rx.lock().await = sdk_control_rx;
        }

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
                    data: serde_json::to_string(&ChatEvent::UserMessage {
                        content: next_msg.clone(),
                    })
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
                shared_sdk_control_rx,
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
            sdk_control_rx,
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
                session.sdk_control_rx.clone(),
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
                data: serde_json::to_string(&ChatEvent::UserMessage {
                    content: message.to_string(),
                })
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
                sdk_control_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Send a permission response (allow/deny) to the Claude CLI subprocess.
    ///
    /// Unlike `send_message`, this does NOT:
    /// - Persist as a user_message event
    /// - Broadcast to WebSocket subscribers
    /// - Trigger a new stream_response
    ///
    /// It sends a JSON control response directly to the CLI subprocess via the
    /// cloned `stdin_tx` sender — **without** taking the `client` Mutex lock.
    ///
    /// This is critical because `stream_response` holds the client lock for the
    /// entire duration of streaming. If we tried to lock the client here, we'd
    /// deadlock: stream_response waits for the permission response to continue,
    /// but we'd wait for stream_response to release the lock.
    ///
    /// The control response format uses the SDK control protocol envelope,
    /// with the inner `response` matching the SDK's `internal_query.rs` format:
    /// ```json
    /// {
    ///   "type": "control_response",
    ///   "response": {
    ///     "subtype": "success",
    ///     "request_id": "<requestId from the control_request>",
    ///     "response": { "allow": true }
    ///   }
    /// }
    /// ```
    /// For deny:
    /// ```json
    /// {
    ///   "type": "control_response",
    ///   "response": {
    ///     "subtype": "success",
    ///     "request_id": "<requestId from the control_request>",
    ///     "response": { "allow": false, "reason": "User denied" }
    ///   }
    /// }
    /// ```
    ///
    /// **IMPORTANT**: Do NOT include `"updatedInput": {}` — the CLI uses that field
    /// to REPLACE the original tool input. An empty `{}` erases the command/input,
    /// causing `"undefined is not an object"` when the CLI tries to execute.
    pub async fn send_permission_response(
        &self,
        session_id: &str,
        request_id: &str,
        allow: bool,
    ) -> Result<()> {
        let (stdin_tx, pending_perm_inputs, events_tx, session_uuid, next_seq) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.last_activity = Instant::now();
            let tx = session.stdin_tx.clone().ok_or_else(|| {
                anyhow!(
                    "No stdin sender for session {} (CLI may not be connected)",
                    session_id
                )
            })?;
            (
                tx,
                session.pending_permission_inputs.clone(),
                session.events_tx.clone(),
                uuid::Uuid::parse_str(session_id).ok(),
                session.next_seq.clone(),
            )
        };

        // Retrieve the original tool input stored when the permission_request arrived.
        // This is needed because the CLI's Zod schema for permission responses requires:
        //   Allow: { behavior: "allow", updatedInput: <record> }
        //   Deny:  { behavior: "deny",  message: <string> }
        // The `updatedInput` field REPLACES the original tool input in the CLI, so we
        // MUST pass back the original input — an empty {} would erase command/file_path/etc.
        let original_input = pending_perm_inputs
            .lock()
            .await
            .remove(request_id)
            .unwrap_or_else(|| serde_json::json!({}));

        let permission_response = if allow {
            serde_json::json!({
                "behavior": "allow",
                "updatedInput": original_input
            })
        } else {
            serde_json::json!({
                "behavior": "deny",
                "message": "User denied the permission request"
            })
        };

        // Wrap in the control_response envelope with subtype + request_id:
        let control_response = serde_json::json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": permission_response
            }
        });

        let json = serde_json::to_string(&control_response)
            .map_err(|e| anyhow!("Failed to serialize control response: {}", e))?;

        info!(
            session_id = %session_id,
            request_id = %request_id,
            allow,
            "Sending permission control response to CLI (via stdin_tx, lock-free)"
        );

        stdin_tx
            .send(json)
            .await
            .map_err(|e| anyhow!("Failed to send permission control response: {}", e))?;

        // Persist and broadcast the permission decision so it survives session reload.
        let decision_event = ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
        };

        // Broadcast to all connected WebSocket clients
        let _ = events_tx.send(decision_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, decision_event.clone());
        }

        // Persist to Neo4j
        if let Some(uuid) = session_uuid {
            let seq = next_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let record = crate::neo4j::models::ChatEventRecord {
                id: uuid::Uuid::new_v4(),
                session_id: uuid,
                seq,
                event_type: "permission_decision".to_string(),
                data: serde_json::to_string(&decision_event).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            if let Err(e) = self.graph.store_chat_events(uuid, vec![record]).await {
                warn!(
                    session_id = %session_id,
                    request_id = %request_id,
                    error = %e,
                    "Failed to persist permission_decision event"
                );
            }
        }

        Ok(())
    }

    /// Change the permission mode of an active CLI session mid-conversation.
    ///
    /// Sends a `set_permission_mode` control request to the Claude CLI subprocess,
    /// updates the in-memory `ActiveSession`, and persists the change to Neo4j.
    ///
    /// **IMPORTANT**: This method sends the control request via the cloned `stdin_tx`
    /// sender — **without** taking the `client` Mutex lock. This is critical because
    /// `stream_response` holds the client lock for the entire duration of streaming.
    /// If we tried to lock the client here, the WS event loop would deadlock:
    /// it awaits the lock while stream_response holds it, so broadcast events
    /// can no longer be forwarded to the frontend.
    pub async fn set_session_permission_mode(&self, session_id: &str, mode: &str) -> Result<()> {
        // Validate mode
        const VALID_MODES: &[&str] = &["default", "acceptEdits", "bypassPermissions", "plan"];
        if !VALID_MODES.contains(&mode) {
            bail!(
                "Invalid permission mode '{}'. Valid modes: {}",
                mode,
                VALID_MODES.join(", ")
            );
        }

        // Get session state and stdin_tx — do NOT extract client (avoids Mutex deadlock)
        let (stdin_tx, old_mode, events_tx) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.last_activity = Instant::now();
            let old_mode = session.permission_mode.clone();
            session.permission_mode = Some(mode.to_string());
            let tx = session.stdin_tx.clone().ok_or_else(|| {
                anyhow!(
                    "No stdin sender for session {} (CLI may not be connected)",
                    session_id
                )
            })?;
            (tx, old_mode, session.events_tx.clone())
        };

        // Build the control_request JSON — same format as InteractiveClient::set_permission_mode
        let control_request = serde_json::json!({
            "type": "control_request",
            "request_id": Uuid::new_v4().to_string(),
            "request": {
                "subtype": "set_permission_mode",
                "mode": mode
            }
        });

        let json = serde_json::to_string(&control_request)
            .map_err(|e| anyhow!("Failed to serialize set_permission_mode request: {}", e))?;

        // Send via stdin_tx (lock-free — bypasses the client Mutex entirely)
        stdin_tx
            .send(json)
            .await
            .map_err(|e| anyhow!("Failed to send set_permission_mode to CLI: {}", e))?;

        info!(
            session_id = %session_id,
            old_mode = ?old_mode,
            new_mode = %mode,
            "Permission mode changed for session (via stdin_tx, lock-free)"
        );

        // Persist to Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self
                .graph
                .update_chat_session_permission_mode(uuid, mode)
                .await
            {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to persist permission mode change to Neo4j (non-fatal)"
                );
            }
        }

        // Broadcast event to WebSocket clients
        let _ = events_tx.send(ChatEvent::PermissionModeChanged {
            mode: mode.to_string(),
        });

        Ok(())
    }

    /// Change the model of an active CLI session mid-conversation.
    ///
    /// Sends a `set_model` control request to the Claude CLI subprocess,
    /// updates the in-memory `ActiveSession`, and broadcasts the change.
    ///
    /// **IMPORTANT**: This method sends the control request via the cloned `stdin_tx`
    /// sender — **without** taking the `client` Mutex lock. This is critical because
    /// `stream_response` holds the client lock for the entire duration of streaming.
    /// If we tried to lock the client here, the WS event loop would deadlock.
    pub async fn set_session_model(&self, session_id: &str, model: &str) -> Result<()> {
        // Get session state and stdin_tx — do NOT extract client (avoids Mutex deadlock)
        let (stdin_tx, old_model, events_tx) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.last_activity = Instant::now();
            let old_model = session.model.clone();
            session.model = Some(model.to_string());
            let tx = session.stdin_tx.clone().ok_or_else(|| {
                anyhow!(
                    "No stdin sender for session {} (CLI may not be connected)",
                    session_id
                )
            })?;
            (tx, old_model, session.events_tx.clone())
        };

        // Build the control_request JSON — same format as InteractiveClient::set_model
        let control_request = serde_json::json!({
            "type": "control_request",
            "request_id": Uuid::new_v4().to_string(),
            "request": {
                "subtype": "set_model",
                "model": model
            }
        });

        let json = serde_json::to_string(&control_request)
            .map_err(|e| anyhow!("Failed to serialize set_model request: {}", e))?;

        // Send via stdin_tx (lock-free — bypasses the client Mutex entirely)
        stdin_tx
            .send(json)
            .await
            .map_err(|e| anyhow!("Failed to send set_model to CLI: {}", e))?;

        info!(
            session_id = %session_id,
            old_model = ?old_model,
            new_model = %model,
            "Model changed for session (via stdin_tx, lock-free)"
        );

        // Broadcast event to WebSocket clients
        let _ = events_tx.send(ChatEvent::ModelChanged {
            model: model.to_string(),
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

        // Create broadcast channel early so CompactionNotifier can use the sender
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);

        // Build PreCompact hook → CompactionNotifier broadcasts ChatEvent::CompactionStarted
        let compaction_hooks = {
            let notifier = CompactionNotifier::new(events_tx.clone());
            let mut hooks = std::collections::HashMap::new();
            hooks.insert(
                "PreCompact".to_string(),
                vec![nexus_claude::HookMatcher {
                    matcher: None,
                    hooks: vec![std::sync::Arc::new(notifier)],
                }],
            );
            hooks
        };

        let options = self
            .build_options(
                &session_node.cwd,
                &session_node.model,
                &system_prompt,
                cli_session_id,
                session_node.permission_mode.as_deref(),
                Some(compaction_hooks),
            )
            .await;

        // Create new InteractiveClient with --resume
        let mut client = InteractiveClient::new(options)
            .map_err(|e| anyhow!("Failed to create InteractiveClient for resume: {}", e))?;

        client
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect resumed InteractiveClient: {}", e))?;

        // Initialize hooks with the CLI (sends PreCompact, etc. registrations).
        // Must be called AFTER connect() and BEFORE take_sdk_control_receiver().
        // Graceful: warn on failure but don't abort the session.
        if let Err(e) = client.initialize_hooks().await {
            warn!(
                session_id = %session_id,
                "Failed to initialize hooks on resume (non-fatal): {}",
                e
            );
        }

        // Take the SDK control receiver ONCE at session resume.
        let sdk_control_rx = client.take_sdk_control_receiver().await;
        let sdk_control_rx = Arc::new(tokio::sync::Mutex::new(sdk_control_rx));

        // Clone stdin sender for lock-free permission responses (see create_session).
        let stdin_tx = client.clone_stdin_sender().await;

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

        // Register as active — cancel old NATS listeners if session was previously active
        let nats_cancel = CancellationToken::new();
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // Cancel stale NATS listeners from a previous resume/create of this session.
            // Without this, each resume_session() spawns 3 new NATS listeners (interrupt,
            // snapshot, RPC) that accumulate — the old ones never stop because the session
            // key still exists in the HashMap (insert replaces the value, not the key).
            // This causes N duplicate stream_response spawns per message, where N is the
            // number of times the session was resumed.
            if let Some(old_session) = sessions.get(session_id) {
                info!(
                    session_id = %session_id,
                    "Cancelling stale NATS listeners for session (resume replacing)"
                );
                old_session.nats_cancel.cancel();
            }
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
                    permission_mode: session_node.permission_mode.clone(),
                    model: Some(session_node.model.clone()),
                    sdk_control_rx: sdk_control_rx.clone(),
                    stdin_tx,
                    nats_cancel: nats_cancel.clone(),
                    pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                        std::collections::HashMap::new(),
                    )),
                },
            );
            interrupt_flag
        };

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            session_id,
            interrupt_flag.clone(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(
            session_id,
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(session_id, self.active_sessions.clone(), nats_cancel);

        // Persist the user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: message.to_string(),
            })
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
                sdk_control_rx,
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

            let has_more = (offset_val + events.len() as i64) < total_count;

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
                            parent_tool_use_id: None,
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

/// Parse a raw SDK control message into a [`ChatEvent::PermissionRequest`] if it is
/// a `can_use_tool` request.  Returns `None` for any other subtype.
///
/// This is the pure-logic core extracted from the `handle_control_msg` closure inside
/// `stream_response`.  Having it as a standalone function makes it directly testable
/// Store a pending permission input in the shared map.
/// Extracted as a standalone async fn so that `tokio::select!` blocks don't
/// struggle with type inference on `Arc<Mutex<HashMap>>` inline locks.
async fn store_pending_perm_input(
    map: &Arc<tokio::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
    id: &str,
    input: &serde_json::Value,
) {
    map.lock().await.insert(id.to_string(), input.clone());
}

/// without needing to spin up a full streaming session.
///
/// The caller is still responsible for:
/// - persisting the event to `events_to_persist`
/// - broadcasting via `emit_chat`
fn parse_permission_control_msg(
    control_msg: &serde_json::Value,
    current_parent: Option<String>,
) -> Option<ChatEvent> {
    // Extract the request data (may be nested under "request")
    let request_data = if control_msg.get("request").is_some() {
        control_msg
            .get("request")
            .cloned()
            .unwrap_or_else(|| control_msg.clone())
    } else {
        control_msg.clone()
    };

    if request_data.get("subtype").and_then(|v| v.as_str()) != Some("can_use_tool") {
        return None;
    }

    let tool_name = request_data
        .get("toolName")
        .or_else(|| request_data.get("tool_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let input = request_data
        .get("input")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let request_id = control_msg
        .get("requestId")
        .or_else(|| control_msg.get("request_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // AskUserQuestion is NOT a permission — it's a user interaction tool.
    // Instead of emitting PermissionRequest (which shows the approval dialog),
    // emit a dedicated AskUserQuestion event so the frontend renders the
    // question widget directly. The caller will auto-allow this request.
    if tool_name == "AskUserQuestion" {
        let questions = input
            .get("questions")
            .cloned()
            .unwrap_or(serde_json::json!([]));
        // Extract the tool_call_id from the control message — this is the tool_use ID
        // that the frontend needs to send the tool_result response back.
        let tool_call_id = request_data
            .get("toolUseId")
            .or_else(|| request_data.get("tool_use_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Some(ChatEvent::AskUserQuestion {
            id: request_id,
            tool_call_id,
            questions,
            input,
            parent_tool_use_id: current_parent,
        });
    }

    Some(ChatEvent::PermissionRequest {
        id: request_id,
        tool: tool_name,
        input,
        parent_tool_use_id: current_parent,
    })
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
            default_model: "claude-sonnet-4-6".into(),
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
            permission: crate::chat::config::PermissionConfig::default(),
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

        assert_eq!(manager.resolve_model(None), "claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn test_build_options_uses_config_permission_default() {
        // Default config uses "default" permission mode (safe-by-default)
        // and pre-approves MCP tools out of the box
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options("/tmp", "claude-opus-4-6", "test prompt", None, None, None)
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
        assert_eq!(opts.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(opts.disallowed_tools.is_empty());
    }

    #[tokio::test]
    async fn test_build_options_uses_config_permission_custom() {
        let state = mock_app_state();
        let mut config = test_config();
        config.permission = crate::chat::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let opts = manager
            .build_options("/tmp", "claude-opus-4-6", "test prompt", None, None, None)
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
        assert_eq!(opts.allowed_tools, vec!["Bash(git *)", "Read"]);
        assert_eq!(opts.disallowed_tools, vec!["Bash(rm -rf *)"]);
    }

    #[tokio::test]
    async fn test_build_options_with_permission_override() {
        // Global config is Default, session overrides to BypassPermissions
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "prompt",
                None,
                Some("bypassPermissions"),
                None,
            )
            .await;
        assert!(matches!(
            opts.permission_mode,
            PermissionMode::BypassPermissions
        ));

        // Without override, falls back to global (Default)
        let opts = manager
            .build_options("/tmp", "claude-opus-4-6", "prompt", None, None, None)
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
    }

    #[tokio::test]
    async fn test_get_permission_config_returns_defaults() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let config = manager.get_permission_config().await;
        assert_eq!(config.mode, "default");
        // MCP tools are pre-approved by default
        assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(config.disallowed_tools.is_empty());
    }

    #[tokio::test]
    async fn test_update_permission_config_changes_mode() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let new_config = crate::chat::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Read".into(), "Bash(git *)".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };
        let updated = manager.update_permission_config(new_config).await.unwrap();
        assert_eq!(updated.mode, "default");
        assert_eq!(updated.allowed_tools, vec!["Read", "Bash(git *)"]);
        assert_eq!(updated.disallowed_tools, vec!["Bash(rm -rf *)"]);

        // Verify getter returns the updated config
        let fetched = manager.get_permission_config().await;
        assert_eq!(fetched.mode, "default");
        assert_eq!(fetched.allowed_tools, vec!["Read", "Bash(git *)"]);
    }

    #[tokio::test]
    async fn test_update_permission_config_rejects_invalid_mode() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let bad_config = crate::chat::config::PermissionConfig {
            mode: "yolo".into(),
            ..Default::default()
        };
        let result = manager.update_permission_config(bad_config).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid permission mode 'yolo'"));
    }

    #[tokio::test]
    async fn test_update_permission_config_affects_build_options() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Initially Default (safe-by-default)
        let opts = manager
            .build_options("/tmp", "claude-opus-4-6", "prompt", None, None, None)
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));

        // Update to BypassPermissions mode at runtime
        manager
            .update_permission_config(crate::chat::config::PermissionConfig {
                mode: "bypassPermissions".into(),
                allowed_tools: vec![],
                disallowed_tools: vec![],
            })
            .await
            .unwrap();

        // New build_options should reflect the update
        let opts = manager
            .build_options("/tmp", "claude-opus-4-6", "prompt", None, None, None)
            .await;
        assert!(matches!(
            opts.permission_mode,
            PermissionMode::BypassPermissions
        ));
        assert!(opts.allowed_tools.is_empty());
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

    #[tokio::test]
    async fn test_build_options_basic() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager
            .build_options(
                "/tmp/project",
                "claude-opus-4-6",
                "System prompt here",
                None,
                None,
                None,
            )
            .await;

        assert_eq!(options.model, Some("claude-opus-4-6".into()));
        assert_eq!(options.cwd, Some(PathBuf::from("/tmp/project")));
        assert!(options.resume.is_none());
        assert!(options.mcp_servers.contains_key("project-orchestrator"));
    }

    #[tokio::test]
    async fn test_build_options_with_resume() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager
            .build_options(
                "/tmp/project",
                "claude-opus-4-6",
                "System prompt",
                Some("cli-session-abc"),
                None,
                None,
            )
            .await;

        assert_eq!(options.resume, Some("cli-session-abc".into()));
    }

    #[tokio::test]
    async fn test_build_options_mcp_server_config() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let options = manager
            .build_options("/tmp", "model", "prompt", None, None, None)
            .await;

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

    #[tokio::test]
    async fn test_build_options_with_hooks() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Build hooks map with a CompactionNotifier
        let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx);
        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let opts = manager
            .build_options("/tmp", "model", "prompt", None, None, Some(hooks))
            .await;

        // Hooks should be configured
        let hook_map = opts.hooks.as_ref().expect("hooks should be Some");
        assert!(hook_map.contains_key("PreCompact"));
        let matchers = &hook_map["PreCompact"];
        assert_eq!(matchers.len(), 1);
        assert_eq!(matchers[0].hooks.len(), 1);
    }

    #[tokio::test]
    async fn test_build_options_without_hooks() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options("/tmp", "model", "prompt", None, None, None)
            .await;

        // No hooks configured
        assert!(opts.hooks.is_none());
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
        assert!(
            matches!(&events[0], ChatEvent::AssistantText { content, .. } if content == "Hello!")
        );
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
            matches!(&events[0], ChatEvent::Thinking { content, .. } if content == "Let me think...")
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
            session_id, duration_ms, cost_usd, subtype, is_error, num_turns, result_text,
        } if session_id == "cli-abc-123"
            && *duration_ms == 5000
            && *cost_usd == Some(0.15)
            && subtype == "success"
            && !is_error
            && *num_turns == Some(3)
            && result_text.is_none()
        ));
    }

    #[test]
    fn test_message_to_events_result_error_max_turns() {
        let msg = Message::Result {
            subtype: "error_max_turns".into(),
            duration_ms: 8000,
            duration_api_ms: 7500,
            is_error: true,
            num_turns: 15,
            session_id: "cli-max-turns".into(),
            total_cost_usd: Some(0.50),
            usage: None,
            result: None,
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            subtype, is_error, num_turns, ..
        } if subtype == "error_max_turns" && *is_error && *num_turns == Some(15)));
    }

    #[test]
    fn test_message_to_events_result_error_during_execution() {
        let msg = Message::Result {
            subtype: "error_during_execution".into(),
            duration_ms: 2000,
            duration_api_ms: 1800,
            is_error: true,
            num_turns: 1,
            session_id: "cli-exec-err".into(),
            total_cost_usd: Some(0.02),
            usage: None,
            result: Some("Process exited with code 1".into()),
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            subtype, is_error, result_text, ..
        } if subtype == "error_during_execution"
            && *is_error
            && result_text.as_deref() == Some("Process exited with code 1")
        ));
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
    fn test_message_to_events_system_init() {
        let msg = Message::System {
            subtype: "init".into(),
            data: serde_json::json!({
                "session_id": "cli-sess-abc",
                "model": "claude-sonnet-4-20250514",
                "tools": ["Bash", "Read", "Write", "Edit"],
                "mcp_servers": [{"name": "po", "status": "connected"}],
                "permissionMode": "default"
            }),
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::SystemInit {
            cli_session_id, model, tools, mcp_servers, permission_mode,
        } if cli_session_id == "cli-sess-abc"
            && model.as_deref() == Some("claude-sonnet-4-20250514")
            && tools.len() == 4
            && mcp_servers.len() == 1
            && permission_mode.as_deref() == Some("default")
        ));
    }

    #[test]
    fn test_message_to_events_system_unknown() {
        // Unknown system subtypes should still be ignored
        let msg = Message::System {
            subtype: "unknown_future_type".into(),
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
        assert!(matches!(&events[0], ChatEvent::StreamDelta { text, .. } if text == "Hello"));
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
                ..
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
            permission_mode: None,
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
                    parent_tool_use_id: None,
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
                    parent_tool_use_id: None,
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
                    parent_tool_use_id: None,
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
            ChatEvent::ToolUse {
                id, tool, input, ..
            } => {
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
                    parent_tool_use_id: None,
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
            permission_mode: None,
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
            permission_mode: None,
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
            permission_mode: None,
            model: None,
            sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_tx: None,
            nats_cancel: CancellationToken::new(),
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
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
                parent_tool_use_id: None,
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"plans": []}),
                is_error: false,
                parent_tool_use_id: None,
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
                parent_tool_use_id: None,
            },
            ChatEvent::StreamingStatus { is_streaming: true },
            ChatEvent::StreamingStatus {
                is_streaming: false,
            },
            ChatEvent::AssistantText {
                content: "Hello world".into(),
                parent_tool_use_id: None,
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
                parent_tool_use_id: None,
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
                parent_tool_use_id: None,
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::PermissionRequest {
                id: "p1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
                parent_tool_use_id: None,
            },
            ChatEvent::InputRequest {
                prompt: "Choose:".into(),
                options: Some(vec!["A".into(), "B".into()]),
                parent_tool_use_id: None,
            },
            ChatEvent::Error {
                message: "Something went wrong".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::Result {
                session_id: "cli-123".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
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
                parent_tool_use_id: None,
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
        manager.spawn_nats_rpc_listener(
            "test-session",
            manager.active_sessions.clone(),
            CancellationToken::new(),
        );
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

    // ====================================================================
    // YAML persistence tests
    // ====================================================================

    #[test]
    fn test_persist_permission_to_yaml_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // Write an existing config.yaml with other sections
        std::fs::write(
            &yaml_path,
            "server:\n  port: 9090\nneo4j:\n  uri: bolt://db:7687\nchat:\n  default_model: claude-sonnet\n",
        )
        .unwrap();

        let perm = super::super::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };

        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // Re-read and verify
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        // Other sections preserved
        assert_eq!(
            doc["server"]["port"].as_u64().unwrap(),
            9090,
            "server.port should be preserved"
        );
        assert_eq!(
            doc["neo4j"]["uri"].as_str().unwrap(),
            "bolt://db:7687",
            "neo4j.uri should be preserved"
        );
        // Existing chat fields preserved
        assert_eq!(
            doc["chat"]["default_model"].as_str().unwrap(),
            "claude-sonnet",
            "chat.default_model should be preserved"
        );
        // Permissions written correctly
        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "default"
        );
        let allowed = doc["chat"]["permissions"]["allowed_tools"]
            .as_sequence()
            .unwrap();
        assert_eq!(allowed.len(), 2);
        assert_eq!(allowed[0].as_str().unwrap(), "Bash(git *)");
        assert_eq!(allowed[1].as_str().unwrap(), "Read");
        let disallowed = doc["chat"]["permissions"]["disallowed_tools"]
            .as_sequence()
            .unwrap();
        assert_eq!(disallowed.len(), 1);
        assert_eq!(disallowed[0].as_str().unwrap(), "Bash(rm -rf *)");
    }

    #[test]
    fn test_persist_permission_to_yaml_no_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // File does not exist yet
        assert!(!yaml_path.exists());

        let perm = super::super::config::PermissionConfig {
            mode: "acceptEdits".into(),
            allowed_tools: vec![],
            disallowed_tools: vec!["Bash(sudo *)".into()],
        };

        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // File should now exist
        assert!(yaml_path.exists());
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "acceptEdits"
        );
        // Empty allowed_tools should be present as empty sequence
        assert!(doc["chat"]["permissions"]["allowed_tools"]
            .as_sequence()
            .unwrap()
            .is_empty());
        assert_eq!(
            doc["chat"]["permissions"]["disallowed_tools"]
                .as_sequence()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_persist_permission_roundtrip_with_config_load() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // Start with a config that has no permissions
        std::fs::write(
            &yaml_path,
            "server:\n  port: 8080\nchat:\n  default_model: claude-opus-4-6\n",
        )
        .unwrap();

        // Persist permissions
        let perm = super::super::config::PermissionConfig {
            mode: "plan".into(),
            allowed_tools: vec!["mcp__project-orchestrator__*".into()],
            disallowed_tools: vec![],
        };
        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // Reload via Config::from_yaml_and_env
        // Clear env vars that would override
        std::env::remove_var("CHAT_PERMISSION_MODE");
        std::env::remove_var("CHAT_ALLOWED_TOOLS");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");

        let config = crate::Config::from_yaml_and_env(Some(&yaml_path)).unwrap();
        let loaded_perm = config
            .chat_permissions
            .expect("chat_permissions should be Some");
        assert_eq!(loaded_perm.mode, "plan");
        assert_eq!(
            loaded_perm.allowed_tools,
            vec!["mcp__project-orchestrator__*"]
        );
        assert!(loaded_perm.disallowed_tools.is_empty());

        // config_yaml_path should be set
        assert_eq!(
            config.config_yaml_path.as_deref(),
            Some(yaml_path.as_path())
        );
    }

    #[test]
    fn test_persist_permission_atomic_no_tmp_leftover() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");
        let tmp_path = dir.path().join("config.yaml.tmp");

        std::fs::write(&yaml_path, "chat: {}\n").unwrap();

        let perm = super::super::config::PermissionConfig::default();
        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // .tmp file should not exist after successful rename
        assert!(
            !tmp_path.exists(),
            "Temporary file should be cleaned up after atomic rename"
        );
        // Original file should still exist with updated content
        assert!(yaml_path.exists());
    }

    #[tokio::test]
    async fn test_update_permission_config_persists_to_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");
        std::fs::write(&yaml_path, "chat:\n  default_model: test\n").unwrap();

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
            .with_config_yaml_path(yaml_path.clone());

        let new_perm = super::super::config::PermissionConfig {
            mode: "acceptEdits".into(),
            allowed_tools: vec!["Read".into()],
            disallowed_tools: vec![],
        };

        let result = manager.update_permission_config(new_perm).await.unwrap();
        assert_eq!(result.mode, "acceptEdits");

        // Verify it was persisted to disk
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();
        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "acceptEdits"
        );
        // Other chat fields preserved
        assert_eq!(doc["chat"]["default_model"].as_str().unwrap(), "test");
    }

    // ── helpers for ActiveSession tests (no Claude CLI required) ────────

    /// Create an ActiveSession backed by a MockTransport.
    /// Does NOT require the Claude CLI to be installed.
    fn mock_active_session(
        is_streaming: bool,
    ) -> (
        ActiveSession,
        nexus_claude::transport::mock::MockTransportHandle,
    ) {
        let (transport, handle) = nexus_claude::transport::mock::MockTransport::pair();
        let client = InteractiveClient::from_transport(transport);
        let (tx, _rx) = broadcast::channel(16);

        let session = ActiveSession {
            events_tx: tx,
            last_activity: Instant::now(),
            cli_session_id: None,
            client: Arc::new(Mutex::new(client)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            memory_manager: None,
            next_seq: Arc::new(AtomicI64::new(1)),
            pending_messages: Arc::new(Mutex::new(VecDeque::new())),
            is_streaming: Arc::new(AtomicBool::new(is_streaming)),
            streaming_text: Arc::new(Mutex::new(String::new())),
            streaming_events: Arc::new(Mutex::new(Vec::new())),
            permission_mode: None,
            model: None,
            sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_tx: None,
            nats_cancel: CancellationToken::new(),
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        };

        (session, handle)
    }

    // ── parse_permission_control_msg tests ──────────────────────────────

    #[test]
    fn test_parse_permission_control_msg_can_use_tool() {
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_abc",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "ls -la"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "should parse can_use_tool");

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id,
                tool,
                input,
                parent_tool_use_id,
            } => {
                assert_eq!(id, "req_abc");
                assert_eq!(tool, "Bash");
                assert_eq!(input["command"], "ls -la");
                assert!(parent_tool_use_id.is_none());
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_with_parent_tool_use_id() {
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_xyz",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "Read",
                "input": {"file_path": "/tmp/test.rs"}
            }
        });

        let event = parse_permission_control_msg(&msg, Some("toolu_parent_123".to_string()));
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id,
                tool,
                input,
                parent_tool_use_id,
            } => {
                assert_eq!(id, "req_xyz");
                assert_eq!(tool, "Read");
                assert_eq!(input["file_path"], "/tmp/test.rs");
                assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_123"));
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ignores_non_permission() {
        // subtype != "can_use_tool"
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_other",
            "request": {
                "subtype": "server_info",
                "data": {"model": "claude-sonnet-4-20250514"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_none(), "non-permission should return None");
    }

    #[test]
    fn test_parse_permission_control_msg_flat_format() {
        // No nested "request" — all fields at top level
        let msg = serde_json::json!({
            "subtype": "can_use_tool",
            "requestId": "flat_001",
            "tool_name": "Write",
            "input": {"file_path": "/tmp/out.txt", "content": "hello"}
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "flat format should parse");

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id, tool, input, ..
            } => {
                assert_eq!(id, "flat_001");
                assert_eq!(tool, "Write");
                assert_eq!(input["content"], "hello");
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_missing_fields_defaults() {
        // Minimal message with just the subtype
        let msg = serde_json::json!({
            "request": {
                "subtype": "can_use_tool"
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id, tool, input, ..
            } => {
                assert_eq!(id, "", "missing request_id defaults to empty");
                assert_eq!(tool, "unknown", "missing tool defaults to 'unknown'");
                assert_eq!(
                    input,
                    serde_json::json!({}),
                    "missing input defaults to empty object"
                );
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_camel_vs_snake_tool_name() {
        // toolName (camelCase) should be preferred
        let msg_camel = serde_json::json!({
            "request_id": "r1",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "CamelTool",
                "input": {}
            }
        });
        let evt = parse_permission_control_msg(&msg_camel, None).unwrap();
        if let ChatEvent::PermissionRequest { tool, .. } = evt {
            assert_eq!(tool, "CamelTool");
        }

        // tool_name (snake_case) fallback
        let msg_snake = serde_json::json!({
            "request_id": "r2",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "SnakeTool",
                "input": {}
            }
        });
        let evt = parse_permission_control_msg(&msg_snake, None).unwrap();
        if let ChatEvent::PermissionRequest { tool, .. } = evt {
            assert_eq!(tool, "SnakeTool");
        }
    }

    #[test]
    fn test_parse_permission_control_msg_request_id_variants() {
        // requestId (camelCase)
        let msg = serde_json::json!({
            "requestId": "camel_id_001",
            "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
        });
        if let Some(ChatEvent::PermissionRequest { id, .. }) =
            parse_permission_control_msg(&msg, None)
        {
            assert_eq!(id, "camel_id_001");
        }

        // request_id (snake_case)
        let msg = serde_json::json!({
            "request_id": "snake_id_002",
            "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
        });
        if let Some(ChatEvent::PermissionRequest { id, .. }) =
            parse_permission_control_msg(&msg, None)
        {
            assert_eq!(id, "snake_id_002");
        }
    }

    // ── parse_permission_control_msg: AskUserQuestion tests ─────────────

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_returns_dedicated_event() {
        let msg = serde_json::json!({
            "type": "control_request",
            "requestId": "req_ask_001",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "toolUseId": "toolu_ask_123",
                "input": {
                    "questions": [{
                        "question": "Which framework?",
                        "header": "Framework",
                        "options": [
                            {"label": "React", "description": "Popular UI library"},
                            {"label": "Vue", "description": "Progressive framework"}
                        ],
                        "multiSelect": false
                    }]
                }
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "AskUserQuestion should parse");

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                id,
                tool_call_id,
                questions,
                input,
                parent_tool_use_id,
            } => {
                assert_eq!(id, "req_ask_001");
                assert_eq!(tool_call_id, "toolu_ask_123");
                assert!(questions.is_array(), "questions should be an array");
                assert_eq!(questions.as_array().unwrap().len(), 1);
                assert_eq!(questions[0]["header"], "Framework");
                assert!(input.get("questions").is_some());
                assert!(parent_tool_use_id.is_none());
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_with_parent() {
        let msg = serde_json::json!({
            "requestId": "req_ask_002",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "toolUseId": "toolu_ask_456",
                "input": {
                    "questions": [{"question": "Choose:", "header": "Q", "options": [{"label": "A"}, {"label": "B"}], "multiSelect": false}]
                }
            }
        });

        let event = parse_permission_control_msg(&msg, Some("toolu_parent_789".to_string()));
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                id,
                parent_tool_use_id,
                ..
            } => {
                assert_eq!(id, "req_ask_002");
                assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_789"));
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_bash_still_returns_permission_request() {
        // Regression: Bash (and all other tools) must still return PermissionRequest
        let msg = serde_json::json!({
            "requestId": "req_bash_001",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "Bash",
                "input": {"command": "echo hello"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest { tool, .. } => {
                assert_eq!(tool, "Bash");
            }
            other => panic!("expected PermissionRequest for Bash, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_missing_questions() {
        // If input doesn't have "questions", should default to empty array
        let msg = serde_json::json!({
            "requestId": "req_ask_003",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "input": {}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                questions,
                tool_call_id,
                ..
            } => {
                assert!(questions.is_array());
                assert_eq!(questions.as_array().unwrap().len(), 0);
                assert_eq!(tool_call_id, "", "missing toolUseId defaults to empty");
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    // ── send_permission_response tests ──────────────────────────────────

    #[tokio::test]
    async fn test_send_permission_response_no_session_errors() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .send_permission_response("nonexistent-session-id", "req-001", true)
            .await;
        assert!(result.is_err(), "should fail for non-existent session");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "error should mention 'not found': {err_msg}"
        );
    }

    // ── interrupt tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_interrupt_no_session_errors() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent-session-id").await;
        // The manager falls back to NATS publish when no local session exists.
        // Without NATS configured, it returns an error or succeeds silently.
        // Either way, it should not panic.
        assert!(
            result.is_ok() || result.is_err(),
            "interrupt on non-existent session should not panic"
        );
    }

    // ── interrupt flag atomicity & latency tests ────────────────────────

    /// Verify that `interrupt()` sets the atomic flag immediately (< 1ms)
    /// without acquiring the InteractiveClient Mutex.
    #[tokio::test]
    async fn test_interrupt_sets_flag_atomically() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        let flag = session.interrupt_flag.clone();

        // Insert session into manager
        let session_id = "test-session-atomic";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        assert!(!flag.load(Ordering::SeqCst), "flag should start false");

        // Measure interrupt latency
        let start = Instant::now();
        manager.interrupt(session_id).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            flag.load(Ordering::SeqCst),
            "flag should be true after interrupt"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "interrupt() took {:?}, expected < 1ms (atomic flag only)",
            elapsed
        );
    }

    /// Verify that `interrupt()` works even when the client Mutex is held
    /// (simulating an active stream_response).
    #[tokio::test]
    async fn test_interrupt_does_not_wait_for_client_lock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(true);
        let flag = session.interrupt_flag.clone();
        let client_lock = session.client.clone();

        let session_id = "test-session-lock";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // Hold the client Mutex (simulating stream_response owning it)
        let _guard = client_lock.lock().await;

        // interrupt() should still complete instantly because it only
        // touches the AtomicBool, not the Mutex
        let start = Instant::now();
        manager.interrupt(session_id).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            flag.load(Ordering::SeqCst),
            "flag should be set even while Mutex is held"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "interrupt() took {:?} even though client Mutex is held — should be < 1ms",
            elapsed
        );
    }

    /// Verify that interrupt works when streaming is active
    /// (is_streaming flag set to true).
    #[tokio::test]
    async fn test_interrupt_during_active_stream() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(true);
        let interrupt_flag = session.interrupt_flag.clone();
        let is_streaming = session.is_streaming.clone();

        let session_id = "test-session-streaming";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        assert!(
            is_streaming.load(Ordering::SeqCst),
            "session should be streaming"
        );
        assert!(
            !interrupt_flag.load(Ordering::SeqCst),
            "interrupt should start false"
        );

        manager.interrupt(session_id).await.unwrap();

        assert!(
            interrupt_flag.load(Ordering::SeqCst),
            "interrupt flag should be set during active stream"
        );
    }

    /// Benchmark: interrupt latency over 100 sessions, each must be < 1ms.
    #[tokio::test]
    async fn test_interrupt_latency_benchmark() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let iterations = 100u32;
        let mut flags = Vec::with_capacity(iterations as usize);

        // Create N sessions
        for i in 0..iterations {
            let (session, _handle) = mock_active_session(true);
            let flag = session.interrupt_flag.clone();
            flags.push(flag);
            manager
                .active_sessions
                .write()
                .await
                .insert(format!("bench-{i}"), session);
        }

        // Interrupt all sessions and measure total time
        let start = Instant::now();
        for i in 0..iterations {
            manager.interrupt(&format!("bench-{i}")).await.unwrap();
        }
        let total = start.elapsed();
        let avg = total / iterations;

        // Verify all flags were set
        for (i, flag) in flags.iter().enumerate() {
            assert!(
                flag.load(Ordering::SeqCst),
                "flag for session bench-{i} should be set"
            );
        }

        assert!(
            avg < Duration::from_millis(1),
            "Average interrupt latency {:?} exceeds 1ms (total {:?} for {iterations})",
            avg,
            total
        );
    }

    /// Verify that double-interrupt doesn't panic and is idempotent.
    #[tokio::test]
    async fn test_double_interrupt_is_idempotent() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        let flag = session.interrupt_flag.clone();

        let session_id = "test-double-interrupt";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // First interrupt
        manager.interrupt(session_id).await.unwrap();
        assert!(flag.load(Ordering::SeqCst));

        // Second interrupt — should not panic or error
        let result = manager.interrupt(session_id).await;
        assert!(result.is_ok(), "second interrupt should succeed");
        assert!(flag.load(Ordering::SeqCst), "flag should remain true");
    }

    // ── send_permission_response with mock transport ────────────────────

    #[tokio::test]
    async fn test_send_permission_response_allow_via_mock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control response
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        // Pre-populate pending_permission_inputs with the original tool input
        let original_input = serde_json::json!({
            "command": "echo \"hello\"",
            "description": "Print hello"
        });
        session
            .pending_permission_inputs
            .lock()
            .await
            .insert("req-allow-001".to_string(), original_input.clone());

        let session_id = "test-perm-allow";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        manager
            .send_permission_response(session_id, "req-allow-001", true)
            .await
            .unwrap();

        // Verify the control response was sent through stdin_tx
        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();

        // Verify full envelope: {"type": "control_response", "response": {"subtype": "success", "request_id": ..., "response": {...}}}
        assert_eq!(sent["type"], "control_response");
        let outer_response = &sent["response"];
        assert_eq!(outer_response["subtype"], "success");
        assert_eq!(outer_response["request_id"], "req-allow-001");
        let inner_response = &outer_response["response"];
        assert_eq!(
            inner_response["behavior"], "allow",
            "inner response should contain behavior: allow"
        );
        assert_eq!(
            inner_response["updatedInput"], original_input,
            "updatedInput should contain the original tool input"
        );
    }

    #[tokio::test]
    async fn test_send_permission_response_deny_via_mock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control response
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        let session_id = "test-perm-deny";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        manager
            .send_permission_response(session_id, "req-deny-002", false)
            .await
            .unwrap();

        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
        assert_eq!(sent["type"], "control_response");
        let outer_response = &sent["response"];
        assert_eq!(outer_response["subtype"], "success");
        assert_eq!(outer_response["request_id"], "req-deny-002");
        let inner_response = &outer_response["response"];
        assert_eq!(
            inner_response["behavior"], "deny",
            "inner response should contain behavior: deny"
        );
        assert!(
            inner_response["message"]
                .as_str()
                .unwrap()
                .contains("denied"),
            "inner response should contain a denial message"
        );
    }

    #[tokio::test]
    async fn test_set_session_model_sends_control_request_via_stdin() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control request
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        // Subscribe to events BEFORE inserting the session
        let mut events_rx = session.events_tx.subscribe();

        let session_id = "test-model-change";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // Change model
        manager
            .set_session_model(session_id, "claude-opus-4-20250514")
            .await
            .unwrap();

        // Verify the control request was sent through stdin_tx
        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
        assert_eq!(sent["type"], "control_request");
        assert!(
            sent["request_id"].as_str().is_some(),
            "should have a request_id"
        );
        let request = &sent["request"];
        assert_eq!(request["subtype"], "set_model");
        assert_eq!(request["model"], "claude-opus-4-20250514");

        // Verify ModelChanged event was broadcast
        let event = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
            .await
            .expect("should receive event within timeout")
            .expect("channel should be open");

        assert_eq!(event.event_type(), "model_changed");
        if let ChatEvent::ModelChanged { model } = event {
            assert_eq!(model, "claude-opus-4-20250514");
        } else {
            panic!("Expected ModelChanged event, got {:?}", event.event_type());
        }

        // Verify the in-memory session was updated
        let sessions = manager.active_sessions.read().await;
        let session = sessions.get(session_id).unwrap();
        assert_eq!(session.model.as_deref(), Some("claude-opus-4-20250514"));
    }

    #[tokio::test]
    async fn test_set_session_model_fails_without_stdin() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        // stdin_tx is None by default

        let session_id = "test-model-no-stdin";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        let result = manager
            .set_session_model(session_id, "claude-opus-4-20250514")
            .await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("No stdin sender"),
            "Error should mention missing stdin sender"
        );
    }

    #[tokio::test]
    async fn test_set_session_model_fails_for_unknown_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .set_session_model("nonexistent-session", "claude-opus-4-20250514")
            .await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "Error should mention session not found"
        );
    }

    // ====================================================================
    // CompactionNotifier
    // ====================================================================

    #[tokio::test]
    async fn test_compaction_notifier_emits_event_on_pre_compact() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx);

        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "test-session".into(),
            transcript_path: "/tmp/transcript".into(),
            cwd: "/test".into(),
            permission_mode: None,
            trigger: "auto".into(),
            custom_instructions: None,
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await;
        assert!(result.is_ok());

        // Verify event was broadcast
        let event = rx.try_recv().unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"
        ));
    }

    #[tokio::test]
    async fn test_compaction_notifier_manual_trigger() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx);

        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "sess-2".into(),
            transcript_path: "/tmp/t".into(),
            cwd: "/".into(),
            permission_mode: Some("default".into()),
            trigger: "manual".into(),
            custom_instructions: Some("Keep API context".into()),
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await.unwrap();

        // Must always return continue=true
        if let nexus_claude::HookJSONOutput::Sync(sync) = result {
            assert_eq!(sync.continue_, Some(true));
        } else {
            panic!("Expected Sync output");
        }

        let event = rx.try_recv().unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionStarted { ref trigger } if trigger == "manual"
        ));
    }

    #[tokio::test]
    async fn test_compaction_notifier_noop_on_other_hooks() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreToolUseHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx);

        let input = HookInput::PreToolUse(PreToolUseHookInput {
            session_id: "s".into(),
            transcript_path: "/t".into(),
            cwd: "/c".into(),
            permission_mode: None,
            tool_name: "Bash".into(),
            tool_input: serde_json::json!({"command": "ls"}),
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, Some("tu1"), &context).await;
        assert!(result.is_ok());

        // No event should be broadcast for non-PreCompact hooks
        assert!(rx.try_recv().is_err());
    }

    // ====================================================================
    // Hook dispatch e2e (CompactionNotifier + dispatch_hook_from_registry)
    // ====================================================================

    /// E2E test: simulates the full hook_callback flow as it happens in stream_response.
    ///
    /// 1. Create a CompactionNotifier with a broadcast channel
    /// 2. Register it via InteractiveClient::initialize_hooks()
    /// 3. Clone the hook_callbacks registry (like stream_response does)
    /// 4. Build a hook_callback JSON message (like the CLI would send)
    /// 5. Dispatch via dispatch_hook_from_registry (lock-free)
    /// 6. Verify CompactionStarted was broadcast
    /// 7. Verify the hook output is continue: true
    #[tokio::test]
    async fn test_compaction_hook_e2e_dispatch_broadcasts_event() {
        use nexus_claude::transport::mock::MockTransport;
        use nexus_claude::{dispatch_hook_from_registry, HookMatcher, InteractiveClient};

        // 1. Create CompactionNotifier backed by a broadcast channel
        let (events_tx, mut events_rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(events_tx);

        // 2. Build hooks map and initialize
        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let (transport, _handle) = MockTransport::pair();
        let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
        client.initialize_hooks().await.unwrap();

        // 3. Clone registry (this is what stream_response does before the select! loop)
        let registry = client.hook_callbacks();

        // 4. Get the callback_id that was generated
        let callback_id = {
            let cbs = registry.read().await;
            assert_eq!(cbs.len(), 1, "Should have exactly one registered callback");
            cbs.keys().next().unwrap().clone()
        };

        // 5. Build a hook_callback control message (simulating what the CLI sends)
        let control_msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req-e2e-001",
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {
                    "hook_event_name": "PreCompact",
                    "session_id": "test-session-e2e",
                    "transcript_path": "/tmp/transcript.json",
                    "cwd": "/home/user/project",
                    "trigger": "auto"
                }
            }
        });

        // 6. Dispatch (lock-free, just like stream_response does)
        let result = dispatch_hook_from_registry(&control_msg, &registry).await;
        assert!(result.is_some(), "dispatch should find the callback");
        let output = result.unwrap();
        assert!(output.is_ok(), "callback should succeed");

        // 7. Verify CompactionStarted was broadcast on the events channel
        let event = events_rx
            .try_recv()
            .expect("Should have received CompactionStarted event");
        assert!(
            matches!(event, ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"),
            "Event should be CompactionStarted with trigger=auto, got: {:?}",
            event
        );

        // 8. Verify output is Sync with continue: true
        match output.unwrap() {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(
                    sync.continue_,
                    Some(true),
                    "Hook should return continue=true"
                );
            }
            other => panic!("Expected Sync output, got: {:?}", other),
        }
    }

    /// Test that build_hook_response_json produces valid JSON that would be sent to CLI.
    /// Verifies the format matches what the CLI expects as a control_response.
    #[tokio::test]
    async fn test_hook_response_sent_to_cli_format() {
        use nexus_claude::transport::mock::MockTransport;
        use nexus_claude::{
            build_hook_response_json, dispatch_hook_from_registry, HookMatcher, InteractiveClient,
        };

        // Setup: CompactionNotifier + initialize + dispatch
        let (events_tx, _events_rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(events_tx);

        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let (transport, _handle) = MockTransport::pair();
        let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
        client.initialize_hooks().await.unwrap();

        let registry = client.hook_callbacks();
        let callback_id = {
            let cbs = registry.read().await;
            cbs.keys().next().unwrap().clone()
        };

        let request_id = "req-response-test-001";
        let control_msg = serde_json::json!({
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {
                    "hook_event_name": "PreCompact",
                    "session_id": "sess-resp",
                    "transcript_path": "/tmp/t.json",
                    "cwd": "/home",
                    "trigger": "manual"
                }
            }
        });

        let result = dispatch_hook_from_registry(&control_msg, &registry)
            .await
            .expect("Should dispatch");

        // Build the response JSON (this is what stream_response sends via stdin_tx)
        let response_json_str = build_hook_response_json(request_id, &result);

        // Parse and verify structure
        let response: serde_json::Value =
            serde_json::from_str(&response_json_str).expect("Should be valid JSON");

        assert_eq!(
            response["type"], "control_response",
            "type must be control_response"
        );

        let resp = &response["response"];
        assert_eq!(resp["subtype"], "success", "subtype must be success");
        assert_eq!(
            resp["request_id"], request_id,
            "request_id must match the original"
        );

        // The inner response should contain the hook output (continue: true)
        let inner = &resp["response"];
        assert_eq!(
            inner["continue"], true,
            "Hook output should have continue=true"
        );
    }
}
