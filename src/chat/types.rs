//! Chat types — request/response/event types for the chat system

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// SpawnedBy — typed origin for sessions spawned by the pipeline or runner
// ---------------------------------------------------------------------------

/// Describes how/why a session was created. Stored as JSON in Neo4j's
/// `spawned_by` property. The enum is **backward-compatible**: the legacy
/// `{"type":"runner", ...}` payloads are parsed via `from_json_str` with
/// a serde fallback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpawnedBy {
    /// Spawned by the classic PlanRunner (wave-based execution).
    Runner {
        run_id: Uuid,
        task_id: Uuid,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<Uuid>,
    },
    /// Spawned by the pipeline orchestrator (WaveExecutor / PipelineEngine).
    Pipeline {
        run_id: Uuid,
        task_id: Uuid,
        wave: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<Uuid>,
    },
    /// Spawned as a gate-retry (quality gate failed, re-running task).
    Gate {
        run_id: Uuid,
        task_id: Uuid,
        gate_name: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<Uuid>,
    },
    /// Spawned by an external trigger or schedule.
    Trigger {
        trigger_id: Uuid,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run_id: Option<Uuid>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<Uuid>,
    },
    /// Sub-conversation spawned by a user or another agent.
    Conversation {
        parent_session_id: Uuid,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
    },
}

impl SpawnedBy {
    /// Serialize to a JSON string suitable for Neo4j storage.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse from a JSON string. Returns `None` for empty/invalid input.
    pub fn from_json_str(s: &str) -> Option<Self> {
        if s.is_empty() {
            return None;
        }
        serde_json::from_str(s).ok()
    }

    /// Extract the parent session ID if present.
    pub fn parent_session_id(&self) -> Option<Uuid> {
        match self {
            Self::Runner {
                parent_session_id, ..
            } => *parent_session_id,
            Self::Pipeline {
                parent_session_id, ..
            } => *parent_session_id,
            Self::Gate {
                parent_session_id, ..
            } => *parent_session_id,
            Self::Trigger { .. } => None,
            Self::Conversation {
                parent_session_id, ..
            } => Some(*parent_session_id),
        }
    }

    /// Return a short label for the spawn type (for display/logging).
    pub fn spawn_type(&self) -> &'static str {
        match self {
            Self::Runner { .. } => "runner",
            Self::Pipeline { .. } => "pipeline",
            Self::Gate { .. } => "gate",
            Self::Trigger { .. } => "trigger",
            Self::Conversation { .. } => "conversation",
        }
    }

    /// Extract run_id if present.
    pub fn run_id(&self) -> Option<Uuid> {
        match self {
            Self::Runner { run_id, .. } => Some(*run_id),
            Self::Pipeline { run_id, .. } => Some(*run_id),
            Self::Gate { run_id, .. } => Some(*run_id),
            Self::Trigger { run_id, .. } => *run_id,
            Self::Conversation { .. } => None,
        }
    }

    /// Extract task_id if present.
    pub fn task_id(&self) -> Option<Uuid> {
        match self {
            Self::Runner { task_id, .. } => Some(*task_id),
            Self::Pipeline { task_id, .. } => Some(*task_id),
            Self::Gate { task_id, .. } => Some(*task_id),
            Self::Trigger { task_id, .. } => *task_id,
            Self::Conversation { .. } => None,
        }
    }
}

/// Runner-specific context passed through `ChatRequest` so that `create_session`
/// can build a runner-specific system prompt instead of the generic PO prompt.
///
/// When this is `Some`, the session is an autonomous code execution agent — the
/// system prompt will instruct Claude to write code, test, and commit without
/// conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerContext {
    /// Git branch the agent must commit on.
    pub git_branch: String,
    /// Files the agent is expected to modify.
    pub affected_files: Vec<String>,
    /// Files the agent must NOT modify (managed by parallel agents).
    pub forbidden_files: Vec<String>,
    /// Activated skill context text (pre-rendered).
    pub skill_context: Option<String>,
    /// Current wave number (for parallel execution awareness).
    pub wave_number: usize,
    /// Number of parallel agents in this wave.
    pub parallel_agents: usize,
    /// Scaffolding level (0-4) for adaptive prompt verbosity.
    pub scaffolding_level: u8,
    /// Frustration score (0.0-1.0) — high = previous attempts failed.
    pub frustration_level: f64,
    /// Task tags (test, refactor, security, etc.) for conditional constraints.
    pub task_tags: Vec<String>,
}

impl RunnerContext {
    /// Convert to the `RunnerPromptContext` used by `build_runner_system_prompt`.
    pub fn to_prompt_context(&self) -> crate::runner::prompt::RunnerPromptContext {
        crate::runner::prompt::RunnerPromptContext {
            git_branch: self.git_branch.clone(),
            task_tags: self.task_tags.clone(),
            affected_files: self.affected_files.clone(),
            forbidden_files: self.forbidden_files.clone(),
            skill_context: self.skill_context.clone(),
            frustration_level: self.frustration_level,
            wave_number: self.wave_number,
            parallel_agents: self.parallel_agents,
            scaffolding_level: self.scaffolding_level,
        }
    }
}

/// Request to send a chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The user's message
    pub message: String,
    /// Session ID to resume (optional — creates new session if None)
    #[serde(default)]
    pub session_id: Option<String>,
    /// Working directory for Claude Code CLI
    pub cwd: String,
    /// Project slug to associate with the session
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Model override (default: from ChatConfig)
    #[serde(default)]
    pub model: Option<String>,
    /// Permission mode override for this session (default: from ChatConfig)
    /// Values: "default", "acceptEdits", "plan", "bypassPermissions"
    #[serde(default)]
    pub permission_mode: Option<String>,
    /// Additional directories to expose to Claude Code CLI (--add-dir)
    #[serde(default)]
    pub add_dirs: Option<Vec<String>>,
    /// Workspace slug — if set, the backend resolves all project root_paths
    /// as add_dirs automatically (mutually exclusive with explicit add_dirs)
    #[serde(default)]
    pub workspace_slug: Option<String>,
    /// Authenticated user claims — injected by the server (not from JSON body).
    /// Used to generate the MCP session token (PO_AUTH_TOKEN).
    #[serde(skip)]
    pub user_claims: Option<crate::auth::jwt::Claims>,
    /// Origin of the session — set by PlanRunner or sub-agent spawner.
    /// Serialized JSON string stored as-is in Neo4j.
    /// Not deserialized from HTTP requests (internal use only).
    #[serde(skip)]
    pub spawned_by: Option<String>,
    /// Task context for sub-agent sessions — provides the task description
    /// to `build_system_prompt` so the routing intent detection can produce
    /// a task-aware system prompt instead of a generic one.
    /// Set by `delegate_task` and `PlanRunner.execute_task`.
    /// Not deserialized from HTTP requests (internal use only).
    #[serde(skip)]
    pub task_context: Option<String>,
    /// Inherited scaffolding level from parent session.
    /// When set, overrides the auto-computed scaffolding level in `build_system_prompt`.
    /// This ensures sub-agents inherit the same prompt complexity as their parent.
    #[serde(skip)]
    pub scaffolding_override: Option<u8>,
    /// Runner-specific context — when set, `create_session` uses a runner system
    /// prompt (autonomous code execution) instead of the generic PO assistant prompt.
    /// Set by `PlanRunner.execute_task()` and `delegate_task()`.
    #[serde(skip)]
    pub runner_context: Option<RunnerContext>,
}

/// Events emitted by the chat system (sent via WebSocket / broadcast)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A user message (emitted so multi-tab clients see it)
    UserMessage { content: String },
    /// A system-generated hint (post-compaction context, guard hints, auto-continue).
    /// NOT a user message — frontends should render this differently (or hide it).
    /// Does NOT increment the session's message_count.
    SystemHint { content: String },
    /// Text content from the assistant
    AssistantText {
        content: String,
        /// When set, this event originated from a sub-agent (sidechain).
        /// The value is the tool_use ID of the parent Task that spawned the agent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Claude is thinking (extended thinking)
    Thinking {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Claude is calling a tool
    ToolUse {
        id: String,
        tool: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Result of a tool call
    ToolResult {
        id: String,
        result: serde_json::Value,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Updated input for a tool_use (emitted when AssistantMessage provides full input
    /// after ContentBlockStart already emitted a ToolUse with empty input)
    ToolUseInputResolved {
        id: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Tool execution was cancelled by user interrupt.
    /// Emitted for each pending ToolUse that did not receive a ToolResult
    /// when the stream is interrupted (e.g., user clicks Stop during sleep 60).
    ToolCancelled {
        /// The tool_use ID that was cancelled
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Claude is asking for permission to use a tool
    PermissionRequest {
        id: String,
        tool: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Claude Code called the AskUserQuestion tool — display the interactive
    /// question widget instead of a permission approval dialog.
    /// The `id` is the SDK control request_id (used for auto-allow response).
    /// The `tool_call_id` is the tool_use ID (used by the frontend to send the
    /// tool_result response back).
    AskUserQuestion {
        /// SDK control request ID (auto-responded with allow)
        id: String,
        /// tool_use call ID for sending the tool_result
        tool_call_id: String,
        /// The questions array from the tool input
        questions: serde_json::Value,
        /// Full tool input (for reference)
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Claude is waiting for user input
    InputRequest {
        prompt: String,
        #[serde(default)]
        options: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Conversation turn completed
    Result {
        session_id: String,
        duration_ms: u64,
        #[serde(default)]
        cost_usd: Option<f64>,
        /// Result subtype: "success", "error_max_turns", "error_during_execution"
        #[serde(default)]
        subtype: String,
        /// Whether the result indicates an error
        #[serde(default)]
        is_error: bool,
        /// Number of turns in the conversation
        #[serde(default, skip_serializing_if = "Option::is_none")]
        num_turns: Option<i32>,
        /// Result text or error message
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result_text: Option<String>,
    },
    /// Streaming text delta (real-time token)
    StreamDelta {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// Streaming status change (broadcast to all connected clients)
    StreamingStatus { is_streaming: bool },
    /// An error occurred
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_use_id: Option<String>,
    },
    /// User's decision on a permission request (allow or deny).
    /// Persisted alongside the original PermissionRequest so the decision
    /// is visible when reloading a conversation.
    PermissionDecision {
        /// The permission request ID this decision answers
        id: String,
        /// Whether the tool was allowed
        allow: bool,
    },
    /// Permission mode was changed mid-session
    PermissionModeChanged { mode: String },
    /// Model was changed mid-session
    ModelChanged { model: String },
    /// Context window compaction is starting (emitted via PreCompact hook)
    CompactionStarted {
        /// Compaction trigger: "auto" or "manual"
        trigger: String,
    },
    /// Metrics emitted after a successful post-compaction context re-injection.
    /// Tracks the quality of the recovery: hint size, build latency, and whether
    /// the agent continued normally (no idle warning within 60s — v1: always true
    /// if injection succeeded).
    CompactionRecovery {
        /// Number of tokens in the re-injected hint (estimated: chars / 3).
        hint_tokens: u32,
        /// Time taken to build the compaction context (ms), including Neo4j queries.
        build_latency_ms: u64,
        /// Whether the agent continued the task correctly after compaction.
        /// V1 simplification: true if injection did not fail. A 60s idle timer
        /// will be added in v2.
        recovery_success: bool,
    },
    /// Context window was compacted (automatic or manual)
    CompactBoundary {
        /// Compaction trigger: "auto" or "manual"
        trigger: String,
        /// Number of tokens before compaction (if available)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pre_tokens: Option<u64>,
    },
    /// Session initialized — metadata from the CLI init system message
    SystemInit {
        /// CLI session ID
        cli_session_id: String,
        /// Model used for this session
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// List of available tool names
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<String>,
        /// Connected MCP servers
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mcp_servers: Vec<serde_json::Value>,
        /// Permission mode for this session
        #[serde(default, skip_serializing_if = "Option::is_none")]
        permission_mode: Option<String>,
    },
    /// Backend auto-continue: emitted when the backend detects error_max_turns
    /// and auto_continue is enabled. Signals that a "Continue" will be sent
    /// automatically after the delay.
    AutoContinue {
        session_id: String,
        /// Delay in milliseconds before the auto-continue message is sent
        delay_ms: u64,
    },
    /// Auto-continue state changed for a session (broadcast to sync all frontends)
    AutoContinueStateChanged { session_id: String, enabled: bool },
    /// Backend is retrying after a transient API error (5xx).
    /// Emitted before each retry attempt so the frontend can show a "Retrying..." indicator.
    Retrying {
        /// Current retry attempt (1-indexed)
        attempt: u32,
        /// Maximum number of retry attempts
        max_attempts: u32,
        /// Delay in milliseconds before this retry
        delay_ms: u64,
        /// The error message that triggered the retry
        error_message: String,
    },
    /// **Out-of-band** event captured between turns by the permanent OOB
    /// listener (see `chat::manager::spawn_oob_listener`).
    ///
    /// These are `Message`s that arrived on the SDK broadcast while no active
    /// `stream_response` was running — typically background tool notifications
    /// (Monitor output, BashOutput from `run_in_background`, sub-Agent
    /// completion events, etc.).
    ///
    /// Frontends should render these distinctly from in-turn events (e.g. a
    /// dedicated badge or icon on the message bubble). Persisted as
    /// `background_output` and does NOT increment `message_count`.
    BackgroundOutput {
        /// Source descriptor — typically the tool name (e.g. "Monitor",
        /// "BashOutput") or "system" for non-tool spontaneous messages.
        source: String,
        /// Human-readable content of the spontaneous message.
        content: String,
        /// ISO-8601 timestamp at which the OOB listener received the event.
        received_at: chrono::DateTime<chrono::Utc>,
        /// Optional correlation ID linking this event to a prior tool_use
        /// (e.g. the background bash ID, the parent task ID).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// A session-level error that requires the user's attention — the CLI
    /// subprocess is gone, the transport is broken, or the session is
    /// otherwise unrecoverable without a fresh `create_session` /
    /// `resume_session` round-trip.
    ///
    /// Distinct from `Error`, which is used for in-stream tool errors
    /// recoverable mid-turn. `SessionError` signals that subsequent
    /// messages on this session_id will silently no-op until the user
    /// (or runner) explicitly restarts the session — frontends should
    /// surface this prominently.
    ///
    /// T9 of plan 9a1684b2 — emitted by `chat::oob_listener` when its
    /// stream returns `None` (broadcast closed because the SDK transport
    /// dropped its sender).
    SessionError {
        /// Short machine-readable reason code (e.g. "subprocess_exited",
        /// "transport_closed", "broadcast_dropped"). Stable across
        /// versions for log filtering.
        reason: String,
        /// Human-readable message safe to show to the end user.
        message: String,
        /// ISO-8601 timestamp at which the error was detected.
        received_at: chrono::DateTime<chrono::Utc>,
    },
}

impl ChatEvent {
    /// Get the event type name (used for WebSocket messages and persistence)
    pub fn event_type(&self) -> &'static str {
        match self {
            ChatEvent::UserMessage { .. } => "user_message",
            ChatEvent::SystemHint { .. } => "system_hint",
            ChatEvent::AssistantText { .. } => "assistant_text",
            ChatEvent::Thinking { .. } => "thinking",
            ChatEvent::ToolUse { .. } => "tool_use",
            ChatEvent::ToolResult { .. } => "tool_result",
            ChatEvent::ToolUseInputResolved { .. } => "tool_use_input_resolved",
            ChatEvent::ToolCancelled { .. } => "tool_cancelled",
            ChatEvent::PermissionRequest { .. } => "permission_request",
            ChatEvent::AskUserQuestion { .. } => "ask_user_question",
            ChatEvent::InputRequest { .. } => "input_request",
            ChatEvent::Result { .. } => "result",
            ChatEvent::StreamDelta { .. } => "stream_delta",
            ChatEvent::StreamingStatus { .. } => "streaming_status",
            ChatEvent::Error { .. } => "error",
            ChatEvent::PermissionDecision { .. } => "permission_decision",
            ChatEvent::PermissionModeChanged { .. } => "permission_mode_changed",
            ChatEvent::ModelChanged { .. } => "model_changed",
            ChatEvent::CompactionStarted { .. } => "compaction_started",
            ChatEvent::CompactionRecovery { .. } => "compaction_recovery",
            ChatEvent::CompactBoundary { .. } => "compact_boundary",
            ChatEvent::SystemInit { .. } => "system_init",
            ChatEvent::AutoContinue { .. } => "auto_continue",
            ChatEvent::AutoContinueStateChanged { .. } => "auto_continue_state_changed",
            ChatEvent::Retrying { .. } => "retrying",
            ChatEvent::BackgroundOutput { .. } => "background_output",
            ChatEvent::SessionError { .. } => "session_error",
        }
    }

    /// Generate a fingerprint string for dedup during mid-stream join.
    ///
    /// Events with unique IDs (ToolUse, ToolResult, PermissionRequest, etc.) use
    /// the ID as fingerprint. Events without IDs (Thinking, AssistantText) use
    /// a hash of their content. Returns `None` for StreamDelta/StreamingStatus
    /// (these are never in the snapshot and never need dedup).
    pub fn fingerprint(&self) -> Option<String> {
        match self {
            // Events with unique IDs — use type+id
            ChatEvent::ToolUse { id, .. } => Some(format!("tool_use:{}", id)),
            ChatEvent::ToolResult { id, .. } => Some(format!("tool_result:{}", id)),
            ChatEvent::ToolUseInputResolved { id, .. } => {
                Some(format!("tool_use_input_resolved:{}", id))
            }
            ChatEvent::ToolCancelled { id, .. } => Some(format!("tool_cancelled:{}", id)),
            ChatEvent::PermissionRequest { id, .. } => Some(format!("permission_request:{}", id)),
            ChatEvent::AskUserQuestion { id, .. } => Some(format!("ask_user_question:{}", id)),
            ChatEvent::PermissionDecision { id, .. } => Some(format!("permission_decision:{}", id)),
            ChatEvent::SystemInit { cli_session_id, .. } => {
                Some(format!("system_init:{}", cli_session_id))
            }

            // Events without unique IDs — use type + content hash
            ChatEvent::Thinking { content, .. } => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                content.hash(&mut hasher);
                Some(format!("thinking:{}", hasher.finish()))
            }
            ChatEvent::AssistantText { content, .. } => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                content.hash(&mut hasher);
                Some(format!("assistant_text:{}", hasher.finish()))
            }
            ChatEvent::UserMessage { content } => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                content.hash(&mut hasher);
                Some(format!("user_message:{}", hasher.finish()))
            }
            ChatEvent::SystemHint { content } => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                content.hash(&mut hasher);
                Some(format!("system_hint:{}", hasher.finish()))
            }
            ChatEvent::Error { message, .. } => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                message.hash(&mut hasher);
                Some(format!("error:{}", hasher.finish()))
            }
            ChatEvent::Result { session_id, .. } => Some(format!("result:{}", session_id)),
            ChatEvent::InputRequest { prompt, .. } => Some(format!("input_request:{}", prompt)),
            ChatEvent::PermissionModeChanged { mode } => {
                Some(format!("permission_mode_changed:{}", mode))
            }
            ChatEvent::ModelChanged { model } => Some(format!("model_changed:{}", model)),
            ChatEvent::CompactionStarted { trigger } => {
                Some(format!("compaction_started:{}", trigger))
            }
            ChatEvent::CompactionRecovery {
                hint_tokens,
                build_latency_ms,
                recovery_success,
            } => Some(format!(
                "compaction_recovery:{}:{}:{}",
                hint_tokens, build_latency_ms, recovery_success
            )),
            ChatEvent::CompactBoundary { trigger, .. } => {
                Some(format!("compact_boundary:{}", trigger))
            }

            ChatEvent::AutoContinue { session_id, .. } => {
                Some(format!("auto_continue:{}", session_id))
            }
            ChatEvent::AutoContinueStateChanged {
                session_id,
                enabled,
            } => Some(format!(
                "auto_continue_state_changed:{}:{}",
                session_id, enabled
            )),

            ChatEvent::Retrying {
                attempt,
                max_attempts,
                ..
            } => Some(format!("retrying:{}:{}", attempt, max_attempts)),

            ChatEvent::BackgroundOutput {
                source,
                received_at,
                correlation_id,
                ..
            } => {
                // (source, received_at, correlation_id) is enough to dedup OOB
                // events: the same tool firing twice at the same instant with
                // the same correlation is the same event. Content is excluded
                // from the fingerprint so a retry that reformats whitespace
                // does not duplicate the entry.
                let corr = correlation_id.as_deref().unwrap_or("-");
                Some(format!(
                    "background_output:{}:{}:{}",
                    source,
                    received_at.timestamp_millis(),
                    corr
                ))
            }

            ChatEvent::SessionError {
                reason,
                received_at,
                ..
            } => {
                // (reason, timestamp) — the same subprocess can only die
                // once at a given instant; replays at distinct moments
                // are distinct events.
                Some(format!(
                    "session_error:{}:{}",
                    reason,
                    received_at.timestamp_millis()
                ))
            }

            // StreamDelta and StreamingStatus are never in the snapshot
            ChatEvent::StreamDelta { .. } | ChatEvent::StreamingStatus { .. } => None,
        }
    }
}

/// Paginated list of chat events (returned by get_session_messages)
#[derive(Debug, Clone)]
pub struct ChatEventPage {
    /// The events in this page (sorted by seq ASC)
    pub events: Vec<crate::neo4j::models::ChatEventRecord>,
    /// Total number of events for this session
    pub total_count: usize,
    /// Whether there are more events beyond this page
    pub has_more: bool,
    /// Current offset
    pub offset: usize,
    /// Page size limit
    pub limit: usize,
}

/// Messages sent from the client to the server (via POST)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// A new user message
    UserMessage { content: String },
    /// Response to a permission request
    PermissionResponse {
        allow: bool,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Response to an input request
    InputResponse { content: String },
}

/// Chat session metadata (persisted in Neo4j)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    /// Internal session ID (UUID)
    pub id: String,
    /// Claude CLI session ID (for --resume)
    #[serde(default)]
    pub cli_session_id: Option<String>,
    /// Associated project slug
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Associated workspace slug (if session spans a workspace)
    #[serde(default)]
    pub workspace_slug: Option<String>,
    /// Working directory
    pub cwd: String,
    /// Session title (auto-generated or user-provided)
    #[serde(default)]
    pub title: Option<String>,
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: String,
    /// Number of messages exchanged
    #[serde(default)]
    pub message_count: i64,
    /// Total cost in USD
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
    /// Nexus conversation ID (for message history)
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Preview text (first user message, truncated)
    #[serde(default)]
    pub preview: Option<String>,
    /// Permission mode override for this session (None = global config)
    #[serde(default)]
    pub permission_mode: Option<String>,
    /// Additional directories exposed to Claude CLI (--add-dir)
    #[serde(default)]
    pub add_dirs: Option<Vec<String>>,
    /// Origin of the session (runner, sub-conversation, or null for normal)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawned_by: Option<serde_json::Value>,
    /// Plans linked to this session (via ASSOCIATED_WITH or AgentExecution)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_plans: Vec<ChatLinkedPlan>,
    /// Tasks linked to this session
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_tasks: Vec<ChatLinkedTask>,
    /// RFCs transitively linked via plans
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_rfcs: Vec<ChatLinkedRfc>,
}

/// A plan linked to a chat session (API response type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLinkedPlan {
    pub id: String,
    pub title: String,
    pub source: String,
}

/// A task linked to a chat session (API response type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLinkedTask {
    pub id: String,
    pub title: String,
    pub source: String,
}

/// An RFC linked to a chat session (API response type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLinkedRfc {
    pub id: String,
    pub title: String,
}

/// Response when creating a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub stream_url: String,
}

// ============================================================================
// Search types
// ============================================================================

/// A single message hit from full-text search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSearchHit {
    /// Message ID in Meilisearch
    pub message_id: String,
    /// Message role (user or assistant)
    pub role: String,
    /// Truncated content snippet
    pub content_snippet: String,
    /// Turn index within the conversation
    pub turn_index: usize,
    /// Unix timestamp of message creation
    pub created_at: i64,
    /// Relevance score
    pub score: f64,
}

/// Search results grouped by session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSearchResult {
    /// Session ID (UUID)
    pub session_id: String,
    /// Session title
    #[serde(default)]
    pub session_title: Option<String>,
    /// Session preview text
    #[serde(default)]
    pub session_preview: Option<String>,
    /// Associated project slug
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Associated workspace slug (if session was started on a workspace)
    #[serde(default)]
    pub workspace_slug: Option<String>,
    /// Nexus conversation ID
    pub conversation_id: String,
    /// Matching messages in this session
    pub hits: Vec<MessageSearchHit>,
    /// Best (highest) score among hits
    pub best_score: f64,
}

/// Truncate text to a maximum length (in characters, not bytes),
/// preserving word boundaries and UTF-8 safety.
pub fn truncate_snippet(text: &str, max_len: usize) -> String {
    // Count characters, not bytes (UTF-8 safe)
    let char_count = text.chars().count();
    if char_count <= max_len {
        return text.to_string();
    }
    // Collect up to max_len characters and find the byte boundary
    let byte_end = text
        .char_indices()
        .nth(max_len)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let truncated = &text[..byte_end];
    // Try to break at last space for cleaner output
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

// ============================================================================
// API error classification (for retry logic)
// ============================================================================

/// Classification of API errors for retry logic.
#[derive(Debug, Clone, PartialEq)]
pub enum ApiErrorKind {
    /// Retryable server-side error (5xx: api_error, overloaded_error)
    Retryable(String),
    /// Non-retryable client-side error (4xx: invalid_request, auth, rate_limit, etc.)
    NonRetryable(String),
}

impl ApiErrorKind {
    /// Returns `true` if the error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, ApiErrorKind::Retryable(_))
    }
}

/// Known Anthropic API error types that are safe to retry.
const RETRYABLE_ERROR_TYPES: &[&str] = &["api_error", "overloaded_error"];

/// Classify an API error message as retryable or non-retryable.
///
/// Parses error messages from the Anthropic API to determine if they should
/// be retried. Handles multiple error formats:
/// - `"API Error: 500 {"type":"error","error":{"type":"api_error",...}}"`
/// - `"Error: {"type":"error","error":{"type":"api_error",...}}"`
/// - Raw JSON: `{"type":"error","error":{"type":"api_error",...}}`
/// - Plain text containing error type keywords
///
/// **Retryable** (5xx): `api_error`, `overloaded_error`
/// **Non-retryable** (4xx): `invalid_request_error`, `authentication_error`,
/// `rate_limit_error`, `not_found_error`, `permission_error`
pub fn classify_api_error(error_message: &str) -> ApiErrorKind {
    // Try to extract JSON from the error message and parse the error type
    if let Some(json_start) = error_message.find('{') {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&error_message[json_start..])
        {
            // Anthropic format: {"type":"error","error":{"type":"api_error","message":"..."}}
            if let Some(error_type) = parsed
                .get("error")
                .and_then(|e| e.get("type"))
                .and_then(|t| t.as_str())
            {
                if RETRYABLE_ERROR_TYPES.contains(&error_type) {
                    return ApiErrorKind::Retryable(error_type.to_string());
                }
                return ApiErrorKind::NonRetryable(error_type.to_string());
            }
        }
    }

    // Fallback: check for known error type strings in the message text
    for retryable in RETRYABLE_ERROR_TYPES {
        if error_message.contains(retryable) {
            return ApiErrorKind::Retryable(retryable.to_string());
        }
    }

    // Check for HTTP 5xx status codes in the message
    for code in &["500 ", "502 ", "503 ", "529 "] {
        if error_message.contains(code) {
            return ApiErrorKind::Retryable("http_5xx".to_string());
        }
    }

    ApiErrorKind::NonRetryable("unknown".to_string())
}

// ============================================================================
// PendingMessage — typed queue entry for pending_messages
// ============================================================================

/// Distinguishes user-originated messages from system-generated hints
/// in the `pending_messages` queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingMessageKind {
    /// Real user message — persisted as `user_message`, increments message_count
    User,
    /// System-generated hint (post-compaction context, guard hint, auto-continue)
    /// — persisted as `system_hint`, does NOT increment message_count
    SystemHint,
    /// Out-of-band background event captured by the OOB listener while idle
    /// (e.g. Monitor notification, BashOutput from `run_in_background`).
    /// Persisted as `background_output`, does NOT increment message_count.
    /// The OOB listener has ALREADY persisted+broadcasted the corresponding
    /// `ChatEvent::BackgroundOutput` before pushing this entry — `stream_response`
    /// must therefore NOT re-persist it (dedup via this kind).
    BackgroundOutput,
}

/// A typed entry in the pending_messages queue.
///
/// Before this, `pending_messages` was `VecDeque<String>` — all messages
/// were treated as user messages, causing system hints (post-compaction
/// context, guard compaction hints) to appear as if the user sent them.
#[derive(Debug, Clone)]
pub struct PendingMessage {
    pub kind: PendingMessageKind,
    pub content: String,
}

impl PartialEq<&str> for PendingMessage {
    fn eq(&self, other: &&str) -> bool {
        self.content == *other
    }
}

impl PendingMessage {
    pub fn user(content: String) -> Self {
        Self {
            kind: PendingMessageKind::User,
            content,
        }
    }

    pub fn system_hint(content: String) -> Self {
        Self {
            kind: PendingMessageKind::SystemHint,
            content,
        }
    }

    /// Construct a `BackgroundOutput` pending entry. The OOB listener uses
    /// this when an event arrives while the session is idle and a new
    /// `stream_response` must be triggered to surface it to the LLM.
    pub fn background_output(content: String) -> Self {
        Self {
            kind: PendingMessageKind::BackgroundOutput,
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionWorkLog — tracks work performed during a stream for continuity
// ---------------------------------------------------------------------------

/// Tracks the work performed during an active stream session.
///
/// Updated in real-time by parsing `ChatEvent::ToolUse` / `ChatEvent::ToolResult`
/// events during `stream_response`. Serves as the source of truth for resumption
/// after compaction or max_turns.
#[derive(Debug, Clone, Default)]
pub struct SessionWorkLog {
    /// Files that were modified (Write/Edit tool_use)
    pub files_modified: HashSet<String>,
    /// Files that were read (Read/Glob tool_use)
    pub files_read: HashSet<String>,
    /// Steps completed (step(update) with status=completed)
    pub steps_completed: Vec<Uuid>,
    /// Step currently in progress (step(update) with status=in_progress)
    pub step_in_progress: Option<Uuid>,
    /// Short summaries of key decisions/actions
    pub decisions_summary: Vec<String>,
    /// Name of the last tool used
    pub last_tool_name: Option<String>,
    /// Total number of tool_use events processed
    pub tool_use_count: u32,
}

/// Serializable snapshot of `SessionWorkLog` for transmission / persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionWorkLogSnapshot {
    pub files_modified: Vec<String>,
    pub files_read: Vec<String>,
    pub steps_completed: Vec<Uuid>,
    pub step_in_progress: Option<Uuid>,
    pub decisions_summary: Vec<String>,
    pub last_tool_name: Option<String>,
    pub tool_use_count: u32,
}

impl SessionWorkLog {
    /// Create a serializable snapshot (sorted for deterministic output).
    pub fn snapshot(&self) -> SessionWorkLogSnapshot {
        let mut files_modified: Vec<String> = self.files_modified.iter().cloned().collect();
        files_modified.sort();
        let mut files_read: Vec<String> = self.files_read.iter().cloned().collect();
        files_read.sort();
        SessionWorkLogSnapshot {
            files_modified,
            files_read,
            steps_completed: self.steps_completed.clone(),
            step_in_progress: self.step_in_progress,
            decisions_summary: self.decisions_summary.clone(),
            last_tool_name: self.last_tool_name.clone(),
            tool_use_count: self.tool_use_count,
        }
    }

    /// Produce a compact markdown summary for injection into continuity prompts.
    pub fn to_summary_markdown(&self) -> String {
        let mut parts = Vec::new();

        if !self.files_modified.is_empty() {
            let mut files: Vec<&String> = self.files_modified.iter().collect();
            files.sort();
            parts.push(format!(
                "**Files modified ({}):** {}",
                files.len(),
                files
                    .iter()
                    .map(|f| format!("`{}`", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if !self.files_read.is_empty() {
            let mut files: Vec<&String> = self.files_read.iter().collect();
            files.sort();
            parts.push(format!(
                "**Files read ({}):** {}",
                files.len(),
                files
                    .iter()
                    .map(|f| format!("`{}`", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if !self.steps_completed.is_empty() {
            parts.push(format!(
                "**Steps completed ({}):** {}",
                self.steps_completed.len(),
                self.steps_completed
                    .iter()
                    .map(|id| format!("`{}`", id))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if let Some(ref step) = self.step_in_progress {
            parts.push(format!("**Step in progress:** `{}`", step));
        }

        if !self.decisions_summary.is_empty() {
            parts.push(format!(
                "**Decisions:** {}",
                self.decisions_summary.join("; ")
            ));
        }

        if let Some(ref tool) = self.last_tool_name {
            parts.push(format!(
                "**Last tool:** {} (total: {})",
                tool, self.tool_use_count
            ));
        }

        if parts.is_empty() {
            return String::new();
        }

        format!("## Session Work Log\n{}", parts.join("\n"))
    }

    /// Record a tool_use event, extracting file paths from known tool inputs.
    pub fn record_tool_use(&mut self, tool_name: &str, input: &serde_json::Value) {
        self.tool_use_count += 1;
        self.last_tool_name = Some(tool_name.to_string());

        match tool_name {
            "Write" | "Edit" | "NotebookEdit" => {
                if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
                    self.files_modified.insert(path.to_string());
                }
            }
            "Read" => {
                if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
                    self.files_read.insert(path.to_string());
                }
            }
            "Glob" => {
                if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
                    self.files_read.insert(path.to_string());
                }
            }
            "Grep" => {
                if let Some(path) = input.get("path").and_then(|v| v.as_str()) {
                    self.files_read.insert(path.to_string());
                }
            }
            "Bash" => {
                // Track bash commands as decisions for context
                if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                    let short = if cmd.len() > 80 { &cmd[..80] } else { cmd };
                    self.decisions_summary.push(format!("bash: {}", short));
                    // Cap decisions to avoid unbounded growth
                    if self.decisions_summary.len() > 20 {
                        self.decisions_summary.remove(0);
                    }
                }
            }
            // MCP step tool — track step completions
            _ if tool_name.contains("step") => {
                self.record_step_update(input);
            }
            _ => {}
        }
    }

    /// Parse step(update) inputs to track step_in_progress / steps_completed.
    fn record_step_update(&mut self, input: &serde_json::Value) {
        let action = input.get("action").and_then(|v| v.as_str());
        let status = input.get("status").and_then(|v| v.as_str());
        let step_id = input
            .get("step_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        if action != Some("update") {
            return;
        }

        if let Some(id) = step_id {
            match status {
                Some("completed") => {
                    if !self.steps_completed.contains(&id) {
                        self.steps_completed.push(id);
                    }
                    if self.step_in_progress == Some(id) {
                        self.step_in_progress = None;
                    }
                }
                Some("in_progress") => {
                    self.step_in_progress = Some(id);
                }
                _ => {}
            }
        }
    }

    /// Reset the log for a new stream turn (keeps accumulated data across
    /// streams within the same session — only clears per-stream transients).
    pub fn reset_for_new_stream(&mut self) {
        // Intentionally does NOT clear files_modified, files_read, steps_completed, etc.
        // Those accumulate across the session lifetime.
        // Only the last_tool_name is reset since it's per-stream.
        self.last_tool_name = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_deserialize() {
        let json = r#"{
            "message": "Hello",
            "cwd": "/tmp",
            "project_slug": "test-project"
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hello");
        assert_eq!(req.cwd, "/tmp");
        assert_eq!(req.project_slug.as_deref(), Some("test-project"));
        assert!(req.session_id.is_none());
        assert!(req.model.is_none());
    }

    #[test]
    fn test_chat_event_serialize() {
        let event = ChatEvent::AssistantText {
            content: "Hello!".into(),
            parent_tool_use_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"assistant_text\""));
        assert!(json.contains("Hello!"));
        // parent_tool_use_id: None should NOT appear in JSON
        assert!(!json.contains("parent_tool_use_id"));
    }

    #[test]
    fn test_chat_event_types() {
        assert_eq!(
            ChatEvent::AssistantText {
                content: "".into(),
                parent_tool_use_id: None,
            }
            .event_type(),
            "assistant_text"
        );
        assert_eq!(
            ChatEvent::Thinking {
                content: "".into(),
                parent_tool_use_id: None,
            }
            .event_type(),
            "thinking"
        );
        assert_eq!(
            ChatEvent::ToolUse {
                id: "".into(),
                tool: "".into(),
                input: serde_json::Value::Null,
                parent_tool_use_id: None,
            }
            .event_type(),
            "tool_use"
        );
        assert_eq!(
            ChatEvent::ToolCancelled {
                id: "".into(),
                parent_tool_use_id: None,
            }
            .event_type(),
            "tool_cancelled"
        );
        assert_eq!(
            ChatEvent::PermissionRequest {
                id: "".into(),
                tool: "".into(),
                input: serde_json::Value::Null,
                parent_tool_use_id: None,
            }
            .event_type(),
            "permission_request"
        );
        assert_eq!(
            ChatEvent::StreamDelta {
                text: "".into(),
                parent_tool_use_id: None,
            }
            .event_type(),
            "stream_delta"
        );
        assert_eq!(
            ChatEvent::Result {
                session_id: "".into(),
                duration_ms: 0,
                cost_usd: None,
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
            }
            .event_type(),
            "result"
        );
    }

    #[test]
    fn test_client_message_deserialize() {
        let json = r#"{"type": "user_message", "content": "Hi"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::UserMessage { content } if content == "Hi"));

        let json = r#"{"type": "permission_response", "allow": true}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(
            msg,
            ClientMessage::PermissionResponse { allow: true, .. }
        ));

        let json = r#"{"type": "input_response", "content": "option B"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::InputResponse { content } if content == "option B"));
    }

    #[test]
    fn test_chat_event_serde_roundtrip_all_variants() {
        let events = vec![
            ChatEvent::AssistantText {
                content: "Hello!".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::ToolUse {
                id: "tu_1".into(),
                tool: "create_plan".into(),
                input: serde_json::json!({"title": "Plan"}),
                parent_tool_use_id: None,
            },
            ChatEvent::ToolResult {
                id: "tu_1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
                parent_tool_use_id: None,
            },
            ChatEvent::ToolResult {
                id: "tu_2".into(),
                result: serde_json::json!("Not found"),
                is_error: true,
                parent_tool_use_id: Some("toolu_parent_1".into()),
            },
            ChatEvent::ToolCancelled {
                id: "tu_3".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::ToolCancelled {
                id: "tu_4".into(),
                parent_tool_use_id: Some("toolu_parent_3".into()),
            },
            ChatEvent::PermissionRequest {
                id: "pr_1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "rm -rf /"}),
                parent_tool_use_id: None,
            },
            ChatEvent::InputRequest {
                prompt: "Which option?".into(),
                options: Some(vec!["A".into(), "B".into()]),
                parent_tool_use_id: None,
            },
            ChatEvent::InputRequest {
                prompt: "Enter value:".into(),
                options: None,
                parent_tool_use_id: Some("toolu_parent_2".into()),
            },
            ChatEvent::Result {
                session_id: "cli-123".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
                subtype: "success".into(),
                is_error: false,
                num_turns: Some(3),
                result_text: None,
            },
            ChatEvent::Result {
                session_id: "cli-456".into(),
                duration_ms: 1000,
                cost_usd: None,
                subtype: "error_max_turns".into(),
                is_error: true,
                num_turns: Some(15),
                result_text: None,
            },
            ChatEvent::Result {
                session_id: "cli-789".into(),
                duration_ms: 2000,
                cost_usd: Some(0.05),
                subtype: "error_during_execution".into(),
                is_error: true,
                num_turns: Some(1),
                result_text: Some("CLI crashed unexpectedly".into()),
            },
            ChatEvent::StreamDelta {
                text: "Hello".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::Error {
                message: "CLI not found".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::PermissionDecision {
                id: "pr_1".into(),
                allow: true,
            },
            ChatEvent::PermissionDecision {
                id: "pr_2".into(),
                allow: false,
            },
            ChatEvent::ModelChanged {
                model: "claude-opus-4-6".into(),
            },
            ChatEvent::CompactionStarted {
                trigger: "auto".into(),
            },
            ChatEvent::CompactionStarted {
                trigger: "manual".into(),
            },
            ChatEvent::CompactionRecovery {
                hint_tokens: 450,
                build_latency_ms: 120,
                recovery_success: true,
            },
            ChatEvent::CompactionRecovery {
                hint_tokens: 0,
                build_latency_ms: 510,
                recovery_success: false,
            },
            ChatEvent::CompactBoundary {
                trigger: "auto".into(),
                pre_tokens: Some(150000),
            },
            ChatEvent::CompactBoundary {
                trigger: "manual".into(),
                pre_tokens: None,
            },
            ChatEvent::SystemInit {
                cli_session_id: "cli-init-123".into(),
                model: Some("claude-sonnet-4-6".into()),
                tools: vec!["Bash".into(), "Read".into(), "Write".into()],
                mcp_servers: vec![serde_json::json!({"name": "po", "status": "connected"})],
                permission_mode: Some("default".into()),
            },
            ChatEvent::SystemInit {
                cli_session_id: "cli-init-456".into(),
                model: None,
                tools: vec![],
                mcp_servers: vec![],
                permission_mode: None,
            },
            ChatEvent::AutoContinue {
                session_id: "sess-auto-1".into(),
                delay_ms: 500,
            },
            ChatEvent::AutoContinueStateChanged {
                session_id: "sess-auto-2".into(),
                enabled: true,
            },
            ChatEvent::AutoContinueStateChanged {
                session_id: "sess-auto-3".into(),
                enabled: false,
            },
            ChatEvent::Retrying {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 1000,
                error_message: "api_error: Internal server error".into(),
            },
            // T9 of plan 9a1684b2 — SessionError variant.
            ChatEvent::SessionError {
                reason: "subprocess_exited".into(),
                message: "The CLI subprocess for this session has exited.".into(),
                received_at: chrono::Utc::now(),
            },
            ChatEvent::SessionError {
                reason: "transport_closed".into(),
                message: "Lost connection to the CLI transport.".into(),
                received_at: chrono::Utc::now(),
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
            // Verify the type tag roundtrips correctly
            assert_eq!(event.event_type(), deserialized.event_type());
        }
    }

    #[test]
    fn test_auto_continue_event_types_and_fingerprints() {
        let ac = ChatEvent::AutoContinue {
            session_id: "sess-1".into(),
            delay_ms: 500,
        };
        assert_eq!(ac.event_type(), "auto_continue");
        assert_eq!(ac.fingerprint(), Some("auto_continue:sess-1".to_string()));

        let acs_on = ChatEvent::AutoContinueStateChanged {
            session_id: "sess-2".into(),
            enabled: true,
        };
        assert_eq!(acs_on.event_type(), "auto_continue_state_changed");
        assert_eq!(
            acs_on.fingerprint(),
            Some("auto_continue_state_changed:sess-2:true".to_string())
        );

        let acs_off = ChatEvent::AutoContinueStateChanged {
            session_id: "sess-2".into(),
            enabled: false,
        };
        assert_eq!(
            acs_off.fingerprint(),
            Some("auto_continue_state_changed:sess-2:false".to_string())
        );
    }

    #[test]
    fn test_auto_continue_event_deserialize_from_json() {
        let json = r#"{"type":"auto_continue","session_id":"s1","delay_ms":500}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type(), "auto_continue");

        let json2 = r#"{"type":"auto_continue_state_changed","session_id":"s1","enabled":true}"#;
        let event2: ChatEvent = serde_json::from_str(json2).unwrap();
        assert_eq!(event2.event_type(), "auto_continue_state_changed");
    }

    #[test]
    fn test_chat_event_deserialize_from_json() {
        // Test deserializing from a known JSON structure
        let json = r#"{"type":"tool_use","id":"t1","tool":"list_plans","input":{}}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::ToolUse { ref tool, .. } if tool == "list_plans"));

        let json = r#"{"type":"error","message":"fail"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::Error { ref message, .. } if message == "fail"));

        let json = r#"{"type":"permission_request","id":"p1","tool":"bash","input":{"cmd":"ls"}}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::PermissionRequest { .. }));

        let json = r#"{"type":"input_request","prompt":"Choose:","options":["A","B"]}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::InputRequest { ref options, .. } if options.is_some()));

        // CompactionStarted with auto trigger
        let json = r#"{"type":"compaction_started","trigger":"auto"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"));
        assert_eq!(event.event_type(), "compaction_started");
        assert_eq!(
            event.fingerprint(),
            Some("compaction_started:auto".to_string())
        );

        // CompactionStarted with manual trigger
        let json = r#"{"type":"compaction_started","trigger":"manual"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::CompactionStarted { ref trigger } if trigger == "manual")
        );

        // CompactionStarted round-trip serialization
        let original = ChatEvent::CompactionStarted {
            trigger: "auto".to_string(),
        };
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ChatEvent = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(
            deserialized,
            ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"
        ));

        // CompactionRecovery — success
        let json = r#"{"type":"compaction_recovery","hint_tokens":450,"build_latency_ms":120,"recovery_success":true}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionRecovery {
                hint_tokens: 450,
                build_latency_ms: 120,
                recovery_success: true
            }
        ));
        assert_eq!(event.event_type(), "compaction_recovery");
        assert_eq!(
            event.fingerprint(),
            Some("compaction_recovery:450:120:true".to_string())
        );

        // CompactionRecovery — failure
        let json = r#"{"type":"compaction_recovery","hint_tokens":0,"build_latency_ms":510,"recovery_success":false}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionRecovery {
                hint_tokens: 0,
                build_latency_ms: 510,
                recovery_success: false
            }
        ));

        // CompactionRecovery round-trip
        let original = ChatEvent::CompactionRecovery {
            hint_tokens: 200,
            build_latency_ms: 80,
            recovery_success: true,
        };
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ChatEvent = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(
            deserialized,
            ChatEvent::CompactionRecovery {
                hint_tokens: 200,
                build_latency_ms: 80,
                recovery_success: true
            }
        ));

        // CompactBoundary with pre_tokens
        let json = r#"{"type":"compact_boundary","trigger":"auto","pre_tokens":150000}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::CompactBoundary { ref trigger, pre_tokens: Some(150000) } if trigger == "auto")
        );

        // CompactBoundary without pre_tokens (backward compat)
        let json = r#"{"type":"compact_boundary","trigger":"manual"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::CompactBoundary { ref trigger, pre_tokens: None } if trigger == "manual")
        );

        // ModelChanged
        let json = r#"{"type":"model_changed","model":"claude-opus-4-6"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::ModelChanged { ref model } if model == "claude-opus-4-6")
        );

        // SystemInit with all fields
        let json = r#"{"type":"system_init","cli_session_id":"sid-1","model":"claude-sonnet-4-6","tools":["Bash","Read"],"mcp_servers":[{"name":"po"}],"permission_mode":"default"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::SystemInit { ref cli_session_id, ref model, .. } if cli_session_id == "sid-1" && model.as_deref() == Some("claude-sonnet-4-6"))
        );

        // SystemInit minimal (backward compat — only required field)
        let json = r#"{"type":"system_init","cli_session_id":"sid-2"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::SystemInit { ref cli_session_id, ref tools, ref mcp_servers, .. } if cli_session_id == "sid-2" && tools.is_empty() && mcp_servers.is_empty())
        );
    }

    #[test]
    fn test_chat_event_parent_tool_use_id_serde() {
        // (1) parent_tool_use_id: None → JSON should NOT contain the field
        let event = ChatEvent::ToolUse {
            id: "t1".into(),
            tool: "Bash".into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("parent_tool_use_id"));

        // (2) parent_tool_use_id: Some → JSON SHOULD contain the field
        let event = ChatEvent::ToolUse {
            id: "t2".into(),
            tool: "Read".into(),
            input: serde_json::json!({"path": "/src/main.rs"}),
            parent_tool_use_id: Some("toolu_abc123".into()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"parent_tool_use_id\":\"toolu_abc123\""));

        // (3) Backward compat: deserialize JSON WITHOUT parent_tool_use_id → None
        let json = r#"{"type":"tool_use","id":"t3","tool":"Grep","input":{}}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        if let ChatEvent::ToolUse {
            parent_tool_use_id, ..
        } = event
        {
            assert!(parent_tool_use_id.is_none());
        } else {
            panic!("Expected ToolUse variant");
        }

        // (4) Deserialize JSON WITH parent_tool_use_id → Some
        let json = r#"{"type":"stream_delta","text":"hello","parent_tool_use_id":"toolu_xyz789"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        if let ChatEvent::StreamDelta {
            text,
            parent_tool_use_id,
        } = event
        {
            assert_eq!(text, "hello");
            assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_xyz789"));
        } else {
            panic!("Expected StreamDelta variant");
        }

        // (5) Roundtrip with parent — serialize then deserialize
        let event = ChatEvent::ToolResult {
            id: "tu_1".into(),
            result: serde_json::json!({"ok": true}),
            is_error: false,
            parent_tool_use_id: Some("toolu_parent".into()),
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        if let ChatEvent::ToolResult {
            parent_tool_use_id, ..
        } = deserialized
        {
            assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent"));
        } else {
            panic!("Expected ToolResult variant");
        }

        // (6) All variant types that have parent_tool_use_id — roundtrip test
        let events_with_parent: Vec<ChatEvent> = vec![
            ChatEvent::AssistantText {
                content: "hi".into(),
                parent_tool_use_id: Some("p1".into()),
            },
            ChatEvent::Thinking {
                content: "hmm".into(),
                parent_tool_use_id: Some("p2".into()),
            },
            ChatEvent::StreamDelta {
                text: "tok".into(),
                parent_tool_use_id: Some("p3".into()),
            },
            ChatEvent::Error {
                message: "err".into(),
                parent_tool_use_id: Some("p4".into()),
            },
            ChatEvent::PermissionRequest {
                id: "pr1".into(),
                tool: "bash".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: Some("p5".into()),
            },
            ChatEvent::InputRequest {
                prompt: "?".into(),
                options: None,
                parent_tool_use_id: Some("p6".into()),
            },
            ChatEvent::ToolUseInputResolved {
                id: "tu1".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: Some("p7".into()),
            },
        ];
        for event in &events_with_parent {
            let json = serde_json::to_string(event).unwrap();
            assert!(
                json.contains("parent_tool_use_id"),
                "Missing parent_tool_use_id in JSON for {:?}",
                event.event_type()
            );
            let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event.event_type(), deserialized.event_type());
        }
    }

    #[test]
    fn test_client_message_serialize_roundtrip() {
        let messages = vec![
            ClientMessage::UserMessage {
                content: "Hello".into(),
            },
            ClientMessage::PermissionResponse {
                allow: true,
                reason: Some("Trusted tool".into()),
            },
            ClientMessage::PermissionResponse {
                allow: false,
                reason: None,
            },
            ClientMessage::InputResponse {
                content: "option A".into(),
            },
        ];

        for msg in &messages {
            let json = serde_json::to_string(msg).unwrap();
            let deserialized: ClientMessage = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn test_create_session_response_serde() {
        let resp = CreateSessionResponse {
            session_id: "abc-123".into(),
            stream_url: "/api/chat/sessions/abc-123/stream".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: CreateSessionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.session_id, "abc-123");
        assert_eq!(deserialized.stream_url, "/api/chat/sessions/abc-123/stream");
    }

    #[test]
    fn test_chat_request_full_fields() {
        let json = r#"{
            "message": "Create a plan",
            "session_id": "existing-session",
            "cwd": "/home/dev/project",
            "project_slug": "my-project",
            "model": "claude-sonnet-4-6"
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Create a plan");
        assert_eq!(req.session_id.as_deref(), Some("existing-session"));
        assert_eq!(req.cwd, "/home/dev/project");
        assert_eq!(req.project_slug.as_deref(), Some("my-project"));
        assert_eq!(req.model.as_deref(), Some("claude-sonnet-4-6"));
    }

    #[test]
    fn test_chat_session_serde_roundtrip() {
        let session = ChatSession {
            id: "test-id".into(),
            cli_session_id: Some("cli-123".into()),
            project_slug: Some("my-project".into()),
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: Some("Test session".into()),
            model: "claude-opus-4-6".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            message_count: 5,
            total_cost_usd: Some(0.15),
            conversation_id: Some("conv-abc-123".into()),
            preview: Some("Hello, can you help me?".into()),
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
            linked_plans: Vec::new(),
            linked_tasks: Vec::new(),
            linked_rfcs: Vec::new(),
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-id");
        assert_eq!(deserialized.cli_session_id.as_deref(), Some("cli-123"));
        assert_eq!(deserialized.model, "claude-opus-4-6");
        assert_eq!(deserialized.message_count, 5);
    }

    // ====================================================================
    // truncate_snippet
    // ====================================================================

    #[test]
    fn test_truncate_snippet_short_text() {
        let text = "Hello, world!";
        assert_eq!(truncate_snippet(text, 50), "Hello, world!");
    }

    #[test]
    fn test_truncate_snippet_exact_length() {
        let text = "abcde";
        assert_eq!(truncate_snippet(text, 5), "abcde");
    }

    #[test]
    fn test_truncate_snippet_long_text_breaks_at_space() {
        let text = "Hello world this is a long text";
        let result = truncate_snippet(text, 15);
        // Should break at word boundary before char 15 and add "..."
        assert!(result.ends_with("..."));
        assert!(result.len() < text.len());
    }

    #[test]
    fn test_truncate_snippet_no_space_to_break() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let result = truncate_snippet(text, 10);
        // No spaces, so truncates at char boundary
        assert!(result.ends_with("..."));
        assert!(result.starts_with("abcdefghij"));
    }

    #[test]
    fn test_truncate_snippet_utf8_multibyte() {
        let text = "Héllo wörld café résumé";
        let result = truncate_snippet(text, 10);
        assert!(result.ends_with("..."));
        // Should not panic on multi-byte chars
    }

    #[test]
    fn test_truncate_snippet_emoji() {
        let text = "🎉🎊🎈🎁🎀🎆🎇✨🧨🎃🎄";
        let result = truncate_snippet(text, 5);
        assert!(result.ends_with("..."));
        // 5 emoji chars + "..."
        assert_eq!(result.chars().filter(|c| *c != '.').count(), 5);
    }

    #[test]
    fn test_truncate_snippet_empty() {
        assert_eq!(truncate_snippet("", 10), "");
    }

    #[test]
    fn test_truncate_snippet_single_char() {
        assert_eq!(truncate_snippet("a", 10), "a");
    }

    // ====================================================================
    // MessageSearchHit & MessageSearchResult serde
    // ====================================================================

    #[test]
    fn test_message_search_hit_serde_roundtrip() {
        let hit = MessageSearchHit {
            message_id: "msg-123".into(),
            role: "user".into(),
            content_snippet: "Hello, help me with...".into(),
            turn_index: 3,
            created_at: 1700000000,
            score: 0.95,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized: MessageSearchHit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.message_id, "msg-123");
        assert_eq!(deserialized.role, "user");
        assert_eq!(deserialized.turn_index, 3);
        assert!((deserialized.score - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_message_search_result_serde_roundtrip() {
        let result = MessageSearchResult {
            session_id: "sess-abc".into(),
            session_title: Some("My session".into()),
            session_preview: Some("First message preview".into()),
            project_slug: Some("my-project".into()),
            workspace_slug: None,
            conversation_id: "conv-xyz".into(),
            hits: vec![MessageSearchHit {
                message_id: "msg-1".into(),
                role: "assistant".into(),
                content_snippet: "Here is the answer...".into(),
                turn_index: 4,
                created_at: 1700000100,
                score: 0.88,
            }],
            best_score: 0.88,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: MessageSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.session_id, "sess-abc");
        assert_eq!(deserialized.session_title.as_deref(), Some("My session"));
        assert_eq!(
            deserialized.session_preview.as_deref(),
            Some("First message preview")
        );
        assert_eq!(deserialized.project_slug.as_deref(), Some("my-project"));
        assert_eq!(deserialized.conversation_id, "conv-xyz");
        assert_eq!(deserialized.hits.len(), 1);
        assert!((deserialized.best_score - 0.88).abs() < f64::EPSILON);
    }

    #[test]
    fn test_message_search_result_minimal_fields() {
        let result = MessageSearchResult {
            session_id: "sess-1".into(),
            session_title: None,
            session_preview: None,
            project_slug: None,
            workspace_slug: None,
            conversation_id: "conv-1".into(),
            hits: vec![],
            best_score: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: MessageSearchResult = serde_json::from_str(&json).unwrap();
        assert!(deserialized.session_title.is_none());
        assert!(deserialized.session_preview.is_none());
        assert!(deserialized.project_slug.is_none());
        assert!(deserialized.hits.is_empty());
    }

    #[test]
    fn test_chat_session_with_preview_field() {
        let json = r#"{
            "id": "test-id",
            "cwd": "/tmp",
            "model": "claude-opus-4-6",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "preview": "Can you help me build a REST API?"
        }"#;
        let session: ChatSession = serde_json::from_str(json).unwrap();
        assert_eq!(
            session.preview.as_deref(),
            Some("Can you help me build a REST API?")
        );
        // Optional fields default correctly
        assert!(session.title.is_none());
        assert!(session.conversation_id.is_none());
        assert_eq!(session.message_count, 0);
    }

    // ====================================================================
    // workspace_slug & add_dirs fields
    // ====================================================================

    #[test]
    fn test_chat_request_with_workspace_and_add_dirs() {
        let json = r#"{
            "message": "Hello",
            "cwd": "/tmp",
            "workspace_slug": "my-workspace",
            "add_dirs": ["/extra/dir1", "/extra/dir2"]
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.workspace_slug.as_deref(), Some("my-workspace"));
        assert_eq!(
            req.add_dirs.as_deref(),
            Some(vec!["/extra/dir1".to_string(), "/extra/dir2".to_string()].as_slice())
        );
    }

    #[test]
    fn test_chat_request_workspace_defaults_to_none() {
        let json = r#"{"message": "Hi", "cwd": "/tmp"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.workspace_slug.is_none());
        assert!(req.add_dirs.is_none());
    }

    #[test]
    fn test_chat_session_with_workspace_and_add_dirs() {
        let session = ChatSession {
            id: "s1".into(),
            cli_session_id: None,
            project_slug: Some("proj".into()),
            workspace_slug: Some("my-ws".into()),
            cwd: "/tmp".into(),
            title: None,
            model: "model".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: Some(vec!["/dir/a".into(), "/dir/b".into()]),
            spawned_by: None,
            linked_plans: Vec::new(),
            linked_tasks: Vec::new(),
            linked_rfcs: Vec::new(),
        };

        let json = serde_json::to_string(&session).unwrap();
        let de: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(de.workspace_slug.as_deref(), Some("my-ws"));
        assert_eq!(de.add_dirs.as_ref().unwrap().len(), 2);
        assert_eq!(de.add_dirs.as_ref().unwrap()[0], "/dir/a");
        assert_eq!(de.add_dirs.as_ref().unwrap()[1], "/dir/b");
    }

    #[test]
    fn test_chat_session_workspace_fields_default() {
        let json = r#"{
            "id": "s2",
            "cwd": "/tmp",
            "model": "m",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z"
        }"#;
        let session: ChatSession = serde_json::from_str(json).unwrap();
        assert!(session.workspace_slug.is_none());
        assert!(session.add_dirs.is_none());
    }

    #[test]
    fn test_message_search_result_with_workspace_slug() {
        let result = MessageSearchResult {
            session_id: "sess-1".into(),
            session_title: None,
            session_preview: None,
            project_slug: Some("proj".into()),
            workspace_slug: Some("ws-slug".into()),
            conversation_id: "conv-1".into(),
            hits: vec![],
            best_score: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let de: MessageSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(de.workspace_slug.as_deref(), Some("ws-slug"));
    }

    #[test]
    fn test_message_search_result_workspace_slug_default() {
        let json = r#"{
            "session_id": "s",
            "conversation_id": "c",
            "hits": [],
            "best_score": 0.0
        }"#;
        let result: MessageSearchResult = serde_json::from_str(json).unwrap();
        assert!(result.workspace_slug.is_none());
    }

    // ====================================================================
    // Retrying event
    // ====================================================================

    #[test]
    fn test_retrying_event_serde_roundtrip() {
        let event = ChatEvent::Retrying {
            attempt: 2,
            max_attempts: 3,
            delay_ms: 2000,
            error_message: "api_error: Internal server error".into(),
        };
        assert_eq!(event.event_type(), "retrying");

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"retrying\""));
        assert!(json.contains("\"attempt\":2"));
        assert!(json.contains("\"max_attempts\":3"));
        assert!(json.contains("\"delay_ms\":2000"));

        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type(), "retrying");
    }

    #[test]
    fn test_retrying_event_deserialize_from_json() {
        let json = r#"{"type":"retrying","attempt":1,"max_attempts":3,"delay_ms":1000,"error_message":"api_error"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type(), "retrying");
        if let ChatEvent::Retrying {
            attempt,
            max_attempts,
            delay_ms,
            ..
        } = event
        {
            assert_eq!(attempt, 1);
            assert_eq!(max_attempts, 3);
            assert_eq!(delay_ms, 1000);
        } else {
            panic!("Expected Retrying variant");
        }
    }

    #[test]
    fn test_retrying_event_fingerprint() {
        let event = ChatEvent::Retrying {
            attempt: 1,
            max_attempts: 3,
            delay_ms: 1000,
            error_message: "err".into(),
        };
        assert_eq!(event.fingerprint(), Some("retrying:1:3".to_string()));
    }

    // ====================================================================
    // classify_api_error
    // ====================================================================

    #[test]
    fn test_classify_api_error_retryable_json() {
        // Standard Anthropic 500 error format
        let msg = r#"API Error: 500 {"type":"error","error":{"type":"api_error","message":"Internal server error"},"request_id":"req_abc"}"#;
        let kind = classify_api_error(msg);
        assert_eq!(kind, ApiErrorKind::Retryable("api_error".into()));
        assert!(kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_overloaded() {
        let msg = r#"API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let kind = classify_api_error(msg);
        assert_eq!(kind, ApiErrorKind::Retryable("overloaded_error".into()));
        assert!(kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_non_retryable_invalid_request() {
        let msg = r#"API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Bad request"}}"#;
        let kind = classify_api_error(msg);
        assert_eq!(
            kind,
            ApiErrorKind::NonRetryable("invalid_request_error".into())
        );
        assert!(!kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_non_retryable_auth() {
        let msg = r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#;
        let kind = classify_api_error(msg);
        assert_eq!(
            kind,
            ApiErrorKind::NonRetryable("authentication_error".into())
        );
        assert!(!kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_non_retryable_rate_limit() {
        let msg = r#"API Error: 429 {"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#;
        let kind = classify_api_error(msg);
        assert_eq!(kind, ApiErrorKind::NonRetryable("rate_limit_error".into()));
        assert!(!kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_non_retryable_not_found() {
        let msg = r#"{"type":"error","error":{"type":"not_found_error","message":"Not found"}}"#;
        let kind = classify_api_error(msg);
        assert_eq!(kind, ApiErrorKind::NonRetryable("not_found_error".into()));
    }

    #[test]
    fn test_classify_api_error_fallback_keyword() {
        // No JSON, but contains the keyword
        let kind = classify_api_error("Something went wrong: api_error occurred");
        assert_eq!(kind, ApiErrorKind::Retryable("api_error".into()));

        let kind = classify_api_error("overloaded_error: try again");
        assert_eq!(kind, ApiErrorKind::Retryable("overloaded_error".into()));
    }

    #[test]
    fn test_classify_api_error_fallback_http_status() {
        let kind = classify_api_error("HTTP 500 Internal Server Error");
        assert_eq!(kind, ApiErrorKind::Retryable("http_5xx".into()));

        let kind = classify_api_error("Error 503 Service Unavailable");
        assert_eq!(kind, ApiErrorKind::Retryable("http_5xx".into()));

        let kind = classify_api_error("Error 529 Overloaded");
        assert_eq!(kind, ApiErrorKind::Retryable("http_5xx".into()));
    }

    #[test]
    fn test_classify_api_error_unknown() {
        let kind = classify_api_error("Something completely unknown happened");
        assert_eq!(kind, ApiErrorKind::NonRetryable("unknown".into()));
        assert!(!kind.is_retryable());
    }

    #[test]
    fn test_classify_api_error_empty_message() {
        let kind = classify_api_error("");
        assert_eq!(kind, ApiErrorKind::NonRetryable("unknown".into()));
    }

    // ====================================================================
    // SessionWorkLog
    // ====================================================================

    #[test]
    fn test_session_work_log_record_write_edit_read() {
        let mut log = SessionWorkLog::default();

        // Write tool
        log.record_tool_use(
            "Write",
            &serde_json::json!({"file_path": "/src/main.rs", "content": "fn main() {}"}),
        );
        assert!(log.files_modified.contains("/src/main.rs"));
        assert_eq!(log.tool_use_count, 1);
        assert_eq!(log.last_tool_name.as_deref(), Some("Write"));

        // Edit tool
        log.record_tool_use(
            "Edit",
            &serde_json::json!({"file_path": "/src/lib.rs", "old_string": "a", "new_string": "b"}),
        );
        assert!(log.files_modified.contains("/src/lib.rs"));
        assert_eq!(log.tool_use_count, 2);

        // Read tool
        log.record_tool_use("Read", &serde_json::json!({"file_path": "/src/types.rs"}));
        assert!(log.files_read.contains("/src/types.rs"));
        assert_eq!(log.tool_use_count, 3);
    }

    #[test]
    fn test_session_work_log_dedup_files() {
        let mut log = SessionWorkLog::default();

        // Same file written twice → only one entry
        log.record_tool_use(
            "Write",
            &serde_json::json!({"file_path": "/src/main.rs", "content": "v1"}),
        );
        log.record_tool_use(
            "Write",
            &serde_json::json!({"file_path": "/src/main.rs", "content": "v2"}),
        );
        assert_eq!(log.files_modified.len(), 1);
        assert_eq!(log.tool_use_count, 2);
    }

    #[test]
    fn test_session_work_log_step_tracking() {
        let mut log = SessionWorkLog::default();
        let step1 = Uuid::new_v4();
        let step2 = Uuid::new_v4();

        // Step in_progress
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step1.to_string(), "status": "in_progress"}),
        );
        assert_eq!(log.step_in_progress, Some(step1));
        assert!(log.steps_completed.is_empty());

        // Step completed
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step1.to_string(), "status": "completed"}),
        );
        assert_eq!(log.step_in_progress, None);
        assert_eq!(log.steps_completed, vec![step1]);

        // Second step in_progress
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step2.to_string(), "status": "in_progress"}),
        );
        assert_eq!(log.step_in_progress, Some(step2));

        // Complete step2
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step2.to_string(), "status": "completed"}),
        );
        assert_eq!(log.steps_completed, vec![step1, step2]);
    }

    #[test]
    fn test_session_work_log_to_summary_markdown() {
        let mut log = SessionWorkLog::default();
        let step1 = Uuid::new_v4();
        let step2 = Uuid::new_v4();

        // Add 3 files modified
        log.record_tool_use("Write", &serde_json::json!({"file_path": "/a.rs"}));
        log.record_tool_use("Edit", &serde_json::json!({"file_path": "/b.rs"}));
        log.record_tool_use("Write", &serde_json::json!({"file_path": "/c.rs"}));

        // Add 2 steps completed
        log.steps_completed.push(step1);
        log.steps_completed.push(step2);

        let md = log.to_summary_markdown();
        assert!(md.contains("## Session Work Log"));
        assert!(md.contains("`/a.rs`"));
        assert!(md.contains("`/b.rs`"));
        assert!(md.contains("`/c.rs`"));
        assert!(md.contains("Files modified (3)"));
        assert!(md.contains(&step1.to_string()));
        assert!(md.contains(&step2.to_string()));
        assert!(md.contains("Steps completed (2)"));
        assert!(md.contains("Last tool:"));
    }

    #[test]
    fn test_session_work_log_empty_summary() {
        let log = SessionWorkLog::default();
        assert!(log.to_summary_markdown().is_empty());
    }

    #[test]
    fn test_session_work_log_snapshot() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use("Write", &serde_json::json!({"file_path": "/z.rs"}));
        log.record_tool_use("Write", &serde_json::json!({"file_path": "/a.rs"}));
        log.record_tool_use("Read", &serde_json::json!({"file_path": "/m.rs"}));

        let snap = log.snapshot();
        // Snapshot files are sorted
        assert_eq!(snap.files_modified, vec!["/a.rs", "/z.rs"]);
        assert_eq!(snap.files_read, vec!["/m.rs"]);
        assert_eq!(snap.tool_use_count, 3);

        // Snapshot is serializable
        let json = serde_json::to_string(&snap).unwrap();
        let deserialized: SessionWorkLogSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.files_modified, snap.files_modified);
    }

    #[test]
    fn test_session_work_log_reset_for_new_stream() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use("Write", &serde_json::json!({"file_path": "/a.rs"}));
        log.record_tool_use("Read", &serde_json::json!({"file_path": "/b.rs"}));
        assert_eq!(log.last_tool_name.as_deref(), Some("Read"));

        log.reset_for_new_stream();

        // Accumulated data persists
        assert!(log.files_modified.contains("/a.rs"));
        assert!(log.files_read.contains("/b.rs"));
        assert_eq!(log.tool_use_count, 2);
        // Per-stream transient is cleared
        assert!(log.last_tool_name.is_none());
    }

    #[test]
    fn test_session_work_log_bash_decisions_capped() {
        let mut log = SessionWorkLog::default();
        for i in 0..25 {
            log.record_tool_use(
                "Bash",
                &serde_json::json!({"command": format!("echo {}", i)}),
            );
        }
        // Decisions capped at 20
        assert_eq!(log.decisions_summary.len(), 20);
        // Most recent should be the last ones
        assert!(log.decisions_summary.last().unwrap().contains("echo 24"));
    }

    #[test]
    fn test_session_work_log_glob_grep_tracking() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use(
            "Glob",
            &serde_json::json!({"path": "/src", "pattern": "*.rs"}),
        );
        log.record_tool_use(
            "Grep",
            &serde_json::json!({"path": "/src/lib.rs", "pattern": "fn main"}),
        );
        assert!(log.files_read.contains("/src"));
        assert!(log.files_read.contains("/src/lib.rs"));
    }

    #[test]
    fn test_session_work_log_notebook_edit_tracking() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use(
            "NotebookEdit",
            &serde_json::json!({"file_path": "/notebooks/analysis.ipynb", "cell_index": 0}),
        );
        assert!(log.files_modified.contains("/notebooks/analysis.ipynb"));
        assert_eq!(log.tool_use_count, 1);
        assert_eq!(log.last_tool_name.as_deref(), Some("NotebookEdit"));
    }

    #[test]
    fn test_session_work_log_step_update_non_update_action_ignored() {
        let mut log = SessionWorkLog::default();
        let step = Uuid::new_v4();
        // action: "list" should be ignored by record_step_update
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "list", "task_id": step.to_string()}),
        );
        assert!(log.steps_completed.is_empty());
        assert!(log.step_in_progress.is_none());
    }

    #[test]
    fn test_session_work_log_step_update_invalid_uuid_ignored() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": "not-a-uuid", "status": "completed"}),
        );
        assert!(log.steps_completed.is_empty());
    }

    #[test]
    fn test_session_work_log_step_completed_dedup() {
        let mut log = SessionWorkLog::default();
        let step = Uuid::new_v4();
        // Complete the same step twice
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step.to_string(), "status": "completed"}),
        );
        log.record_tool_use(
            "mcp__project-orchestrator__step",
            &serde_json::json!({"action": "update", "step_id": step.to_string(), "status": "completed"}),
        );
        assert_eq!(log.steps_completed.len(), 1);
    }

    #[test]
    fn test_session_work_log_summary_with_step_in_progress_and_decisions() {
        let mut log = SessionWorkLog::default();
        let step = Uuid::new_v4();
        log.step_in_progress = Some(step);
        log.decisions_summary
            .push("Use trait-based approach".to_string());
        log.last_tool_name = Some("Edit".to_string());
        log.tool_use_count = 5;

        let md = log.to_summary_markdown();
        assert!(md.contains("## Session Work Log"));
        assert!(md.contains("Step in progress:"));
        assert!(md.contains(&step.to_string()));
        assert!(md.contains("Decisions:"));
        assert!(md.contains("Use trait-based approach"));
        assert!(
            md.contains("Last tool:") && md.contains("Edit") && md.contains("total: 5"),
            "Missing last tool info in: {}",
            md
        );
    }

    #[test]
    fn test_session_work_log_bash_truncation_at_80_chars() {
        let mut log = SessionWorkLog::default();
        let long_cmd = "a".repeat(120);
        log.record_tool_use("Bash", &serde_json::json!({"command": long_cmd}));
        assert_eq!(log.decisions_summary.len(), 1);
        // "bash: " prefix (6 chars) + 80 chars of command = 86 total
        assert!(
            log.decisions_summary[0].len() <= 86,
            "Decision should be truncated, got len {}",
            log.decisions_summary[0].len()
        );
    }

    #[test]
    fn test_session_work_log_unknown_tool_ignored() {
        let mut log = SessionWorkLog::default();
        log.record_tool_use("SomeRandomTool", &serde_json::json!({"whatever": "value"}));
        assert!(log.files_modified.is_empty());
        assert!(log.files_read.is_empty());
        assert!(log.decisions_summary.is_empty());
        assert_eq!(log.tool_use_count, 1); // still counted
        assert_eq!(log.last_tool_name.as_deref(), Some("SomeRandomTool"));
    }

    #[test]
    fn test_session_work_log_tool_missing_file_path() {
        let mut log = SessionWorkLog::default();
        // Write without file_path key
        log.record_tool_use("Write", &serde_json::json!({"content": "hello"}));
        assert!(log.files_modified.is_empty());
        assert_eq!(log.tool_use_count, 1);
    }

    #[test]
    fn test_background_output_event_roundtrip() {
        let received_at = chrono::DateTime::parse_from_rfc3339("2026-04-30T13:45:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let event = ChatEvent::BackgroundOutput {
            source: "Monitor".into(),
            content: "[epoch 50] cos=0.871".into(),
            received_at,
            correlation_id: Some("bg-abc123".into()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"background_output\""));
        assert!(json.contains("\"source\":\"Monitor\""));
        assert!(json.contains("\"correlation_id\":\"bg-abc123\""));

        let parsed: ChatEvent = serde_json::from_str(&json).unwrap();
        match parsed {
            ChatEvent::BackgroundOutput {
                source,
                content,
                correlation_id,
                ..
            } => {
                assert_eq!(source, "Monitor");
                assert_eq!(content, "[epoch 50] cos=0.871");
                assert_eq!(correlation_id.as_deref(), Some("bg-abc123"));
            }
            other => panic!("expected BackgroundOutput, got {:?}", other),
        }

        assert_eq!(event.event_type(), "background_output");
        let fp = event.fingerprint().expect("BackgroundOutput must have a fingerprint");
        assert!(fp.starts_with("background_output:Monitor:"));
        assert!(fp.ends_with(":bg-abc123"));
    }

    #[test]
    fn test_background_output_omits_correlation_id_when_none() {
        let event = ChatEvent::BackgroundOutput {
            source: "BashOutput".into(),
            content: "build complete".into(),
            received_at: chrono::Utc::now(),
            correlation_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        // None correlation_id should be skipped, not serialized as null
        assert!(!json.contains("correlation_id"));
    }

    #[test]
    fn test_pending_message_background_output() {
        let msg = PendingMessage::background_output("monitor: epoch 100".into());
        assert_eq!(msg.kind, PendingMessageKind::BackgroundOutput);
        assert_eq!(msg.content, "monitor: epoch 100");
    }
}
