//! Chat types — request/response/event types for the chat system

use serde::{Deserialize, Serialize};

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
}

/// Events emitted by the chat system (sent via WebSocket / broadcast)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A user message (emitted so multi-tab clients see it)
    UserMessage { content: String },
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
}

impl ChatEvent {
    /// Get the event type name (used for WebSocket messages and persistence)
    pub fn event_type(&self) -> &'static str {
        match self {
            ChatEvent::UserMessage { .. } => "user_message",
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
            ChatEvent::CompactBoundary { .. } => "compact_boundary",
            ChatEvent::SystemInit { .. } => "system_init",
            ChatEvent::AutoContinue { .. } => "auto_continue",
            ChatEvent::AutoContinueStateChanged { .. } => "auto_continue_state_changed",
            ChatEvent::Retrying { .. } => "retrying",
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
                model: "claude-opus-4-20250514".into(),
            },
            ChatEvent::CompactionStarted {
                trigger: "auto".into(),
            },
            ChatEvent::CompactionStarted {
                trigger: "manual".into(),
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
                model: Some("claude-sonnet-4-20250514".into()),
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
        let json = r#"{"type":"model_changed","model":"claude-opus-4-20250514"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::ModelChanged { ref model } if model == "claude-opus-4-20250514")
        );

        // SystemInit with all fields
        let json = r#"{"type":"system_init","cli_session_id":"sid-1","model":"claude-sonnet-4-20250514","tools":["Bash","Read"],"mcp_servers":[{"name":"po"}],"permission_mode":"default"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(
            matches!(event, ChatEvent::SystemInit { ref cli_session_id, ref model, .. } if cli_session_id == "sid-1" && model.as_deref() == Some("claude-sonnet-4-20250514"))
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
            "model": "claude-sonnet-4-20250514"
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Create a plan");
        assert_eq!(req.session_id.as_deref(), Some("existing-session"));
        assert_eq!(req.cwd, "/home/dev/project");
        assert_eq!(req.project_slug.as_deref(), Some("my-project"));
        assert_eq!(req.model.as_deref(), Some("claude-sonnet-4-20250514"));
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
        let json =
            r#"{"type":"retrying","attempt":1,"max_attempts":3,"delay_ms":1000,"error_message":"api_error"}"#;
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
        assert_eq!(
            kind,
            ApiErrorKind::NonRetryable("rate_limit_error".into())
        );
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
}
