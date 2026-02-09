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
}

/// Events emitted by the chat system (sent via WebSocket / broadcast)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A user message (emitted so multi-tab clients see it)
    UserMessage { content: String },
    /// Text content from the assistant
    AssistantText { content: String },
    /// Claude is thinking (extended thinking)
    Thinking { content: String },
    /// Claude is calling a tool
    ToolUse {
        id: String,
        tool: String,
        input: serde_json::Value,
    },
    /// Result of a tool call
    ToolResult {
        id: String,
        result: serde_json::Value,
        #[serde(default)]
        is_error: bool,
    },
    /// Claude is asking for permission to use a tool
    PermissionRequest {
        id: String,
        tool: String,
        input: serde_json::Value,
    },
    /// Claude is waiting for user input
    InputRequest {
        prompt: String,
        #[serde(default)]
        options: Option<Vec<String>>,
    },
    /// Conversation turn completed
    Result {
        session_id: String,
        duration_ms: u64,
        #[serde(default)]
        cost_usd: Option<f64>,
    },
    /// Streaming text delta (real-time token)
    StreamDelta { text: String },
    /// Streaming status change (broadcast to all connected clients)
    StreamingStatus { is_streaming: bool },
    /// An error occurred
    Error { message: String },
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
            ChatEvent::PermissionRequest { .. } => "permission_request",
            ChatEvent::InputRequest { .. } => "input_request",
            ChatEvent::Result { .. } => "result",
            ChatEvent::StreamDelta { .. } => "stream_delta",
            ChatEvent::StreamingStatus { .. } => "streaming_status",
            ChatEvent::Error { .. } => "error",
        }
    }
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
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"assistant_text\""));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_event_types() {
        assert_eq!(
            ChatEvent::AssistantText { content: "".into() }.event_type(),
            "assistant_text"
        );
        assert_eq!(
            ChatEvent::Thinking { content: "".into() }.event_type(),
            "thinking"
        );
        assert_eq!(
            ChatEvent::ToolUse {
                id: "".into(),
                tool: "".into(),
                input: serde_json::Value::Null
            }
            .event_type(),
            "tool_use"
        );
        assert_eq!(
            ChatEvent::PermissionRequest {
                id: "".into(),
                tool: "".into(),
                input: serde_json::Value::Null
            }
            .event_type(),
            "permission_request"
        );
        assert_eq!(
            ChatEvent::StreamDelta { text: "".into() }.event_type(),
            "stream_delta"
        );
        assert_eq!(
            ChatEvent::Result {
                session_id: "".into(),
                duration_ms: 0,
                cost_usd: None
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
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
            },
            ChatEvent::ToolUse {
                id: "tu_1".into(),
                tool: "create_plan".into(),
                input: serde_json::json!({"title": "Plan"}),
            },
            ChatEvent::ToolResult {
                id: "tu_1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
            },
            ChatEvent::ToolResult {
                id: "tu_2".into(),
                result: serde_json::json!("Not found"),
                is_error: true,
            },
            ChatEvent::PermissionRequest {
                id: "pr_1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "rm -rf /"}),
            },
            ChatEvent::InputRequest {
                prompt: "Which option?".into(),
                options: Some(vec!["A".into(), "B".into()]),
            },
            ChatEvent::InputRequest {
                prompt: "Enter value:".into(),
                options: None,
            },
            ChatEvent::Result {
                session_id: "cli-123".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
            },
            ChatEvent::Result {
                session_id: "cli-456".into(),
                duration_ms: 1000,
                cost_usd: None,
            },
            ChatEvent::StreamDelta {
                text: "Hello".into(),
            },
            ChatEvent::Error {
                message: "CLI not found".into(),
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
    fn test_chat_event_deserialize_from_json() {
        // Test deserializing from a known JSON structure
        let json = r#"{"type":"tool_use","id":"t1","tool":"list_plans","input":{}}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::ToolUse { ref tool, .. } if tool == "list_plans"));

        let json = r#"{"type":"error","message":"fail"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::Error { ref message } if message == "fail"));

        let json = r#"{"type":"permission_request","id":"p1","tool":"bash","input":{"cmd":"ls"}}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::PermissionRequest { .. }));

        let json = r#"{"type":"input_request","prompt":"Choose:","options":["A","B"]}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ChatEvent::InputRequest { ref options, .. } if options.is_some()));
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
            cwd: "/tmp".into(),
            title: Some("Test session".into()),
            model: "claude-opus-4-6".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            message_count: 5,
            total_cost_usd: Some(0.15),
            conversation_id: Some("conv-abc-123".into()),
            preview: Some("Hello, can you help me?".into()),
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-id");
        assert_eq!(deserialized.cli_session_id.as_deref(), Some("cli-123"));
        assert_eq!(deserialized.model, "claude-opus-4-6");
        assert_eq!(deserialized.message_count, 5);
    }
}
