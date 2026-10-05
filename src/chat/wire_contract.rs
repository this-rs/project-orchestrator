//! Field-level wire contract of the chat WebSocket (decision A44).
//!
//! Test-only module. For every frame that crosses `/ws/chat/{session_id}` it
//! holds two example frames (`full`: every optional field set, `minimal`:
//! every optional field absent) and derives from them a per-field contract
//! (`type`, `required`, `nullable`). The result is written to
//! `docs/api/chat-contract/`, which the frontend copies as is (checked with
//! `SHA256SUMS`) and compares to its TypeScript types.
//!
//! Three families of frames:
//! - server events: every [`ChatEvent`] variant, serialized by serde;
//! - client messages: every [`WsChatClientMessage`] variant (it is only
//!   `Deserialize`, so the examples are JSON literals checked by parsing);
//! - control frames: the `json!` frames the WS handler builds by hand, which
//!   are not `ChatEvent`s. Those are static copies — nothing ties them to the
//!   handler code but the source function named next to each of them.
//!
//! Regenerate after any change of the wire:
//! `UPDATE_CHAT_CONTRACT=1 cargo test --lib chat::wire_contract`

use crate::api::ws_chat_handler::WsChatClientMessage;
use crate::chat::message_attachments::MessageAttachment;
use crate::chat::types::{BackgroundTaskInfo, BackgroundTaskKind, ChatEvent, PendingQueueEntry};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Bumped by hand when the SHAPE of the fixture files changes (not when an
/// event gains a field: that is what the fixtures themselves record).
const CONTRACT_VERSION: u32 = 1;

/// Directory of the committed fixtures, relative to the crate root.
const FIXTURE_DIR: &str = "docs/api/chat-contract";

const SERVER_EVENTS_FILE: &str = "server-events.json";
const CLIENT_MESSAGES_FILE: &str = "client-messages.json";
const CONTROL_FRAMES_FILE: &str = "control-frames.json";
const CHECKSUMS_FILE: &str = "SHA256SUMS";

const UPDATE_ENV: &str = "UPDATE_CHAT_CONTRACT";
const UPDATE_COMMAND: &str = "UPDATE_CHAT_CONTRACT=1 cargo test --lib chat::wire_contract";

/// Every `ChatEvent` tag the examples must cover. Adding a variant means
/// adding its tag here AND its examples in [`server_examples`].
const EXPECTED_SERVER_TAGS: [&str; 33] = [
    "active_tasks_update",
    "ask_user_question",
    "assistant_text",
    "auto_continue",
    "auto_continue_state_changed",
    "background_output",
    "compact_boundary",
    "compaction_recovery",
    "compaction_started",
    "error",
    "model_changed",
    "pending_queue",
    "permission_decision",
    "permission_mode_changed",
    "permission_request",
    "result",
    "retrying",
    "secret_request",
    "secret_request_resolved",
    "session_closed",
    "session_error",
    "stream_delta",
    "streaming_status",
    "system_hint",
    "system_init",
    "thinking",
    "tool_cancelled",
    "tool_result",
    "tool_use",
    "tool_use_input_resolved",
    "tools_cancelled",
    "user_message",
    "workflow",
];

/// Every `WsChatClientMessage` tag the examples must cover.
const EXPECTED_CLIENT_TAGS: [&str; 9] = [
    "cancel_tools",
    "input_response",
    "interrupt",
    "permission_response",
    "queue_op",
    "set_auto_continue",
    "set_model",
    "set_permission_mode",
    "user_message",
];

// ============================================================================
// Variant tags — exhaustive on purpose
// ============================================================================

/// Wire tag of a `ChatEvent` variant.
///
/// No `_` arm: a new variant stops the build HERE, which is the reminder to
/// add its examples to [`server_examples`] and its tag to
/// [`EXPECTED_SERVER_TAGS`].
fn variant_tag(e: &ChatEvent) -> &'static str {
    match e {
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
        ChatEvent::Result { .. } => "result",
        ChatEvent::StreamDelta { .. } => "stream_delta",
        ChatEvent::StreamingStatus { .. } => "streaming_status",
        ChatEvent::PendingQueue { .. } => "pending_queue",
        ChatEvent::Error { .. } => "error",
        ChatEvent::PermissionDecision { .. } => "permission_decision",
        ChatEvent::PermissionModeChanged { .. } => "permission_mode_changed",
        ChatEvent::ModelChanged { .. } => "model_changed",
        ChatEvent::CompactionStarted { .. } => "compaction_started",
        ChatEvent::CompactionRecovery { .. } => "compaction_recovery",
        ChatEvent::CompactBoundary { .. } => "compact_boundary",
        ChatEvent::SystemInit { .. } => "system_init",
        ChatEvent::Workflow { .. } => "workflow",
        ChatEvent::AutoContinue { .. } => "auto_continue",
        ChatEvent::AutoContinueStateChanged { .. } => "auto_continue_state_changed",
        ChatEvent::Retrying { .. } => "retrying",
        ChatEvent::BackgroundOutput { .. } => "background_output",
        ChatEvent::ToolsCancelled { .. } => "tools_cancelled",
        ChatEvent::SessionError { .. } => "session_error",
        ChatEvent::ActiveTasksUpdate { .. } => "active_tasks_update",
        ChatEvent::SecretRequest { .. } => "secret_request",
        ChatEvent::SecretRequestResolved { .. } => "secret_request_resolved",
        ChatEvent::SessionClosed { .. } => "session_closed",
    }
}

/// Wire tag of a client message variant. No `_` arm, same reason as
/// [`variant_tag`]: a new variant needs examples in [`client_examples`].
fn client_tag(m: &WsChatClientMessage) -> &'static str {
    match m {
        WsChatClientMessage::UserMessage { .. } => "user_message",
        WsChatClientMessage::QueueOp { .. } => "queue_op",
        WsChatClientMessage::Interrupt => "interrupt",
        WsChatClientMessage::PermissionResponse { .. } => "permission_response",
        WsChatClientMessage::InputResponse { .. } => "input_response",
        WsChatClientMessage::SetPermissionMode { .. } => "set_permission_mode",
        WsChatClientMessage::SetModel { .. } => "set_model",
        WsChatClientMessage::SetAutoContinue { .. } => "set_auto_continue",
        WsChatClientMessage::CancelTools => "cancel_tools",
    }
}

// ============================================================================
// Stable example values (no clock, no randomness)
// ============================================================================

fn s(text: &str) -> String {
    text.to_string()
}

fn ts(rfc3339: &str) -> chrono::DateTime<chrono::Utc> {
    rfc3339
        .parse::<chrono::DateTime<chrono::Utc>>()
        .expect("constant RFC 3339 timestamp")
}

fn fixed_uuid(text: &str) -> Uuid {
    Uuid::parse_str(text).expect("constant UUID")
}

const SESSION_ID: &str = "5b0c7c1e-2f4a-4d6b-9a3e-1c2d3e4f5a6b";
const PARENT_TOOL_USE_ID: &str = "toolu_01ParentTask0000000000";
const TOOL_USE_ID: &str = "toolu_01Example0000000000000";
const QUEUED_MESSAGE_ID: &str = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d";
const DOCUMENT_ID: &str = "9f8e7d6c-5b4a-4c3d-9e2f-1a0b9c8d7e6f";

/// The two example frames of one `ChatEvent` variant.
struct ServerExample {
    full: ChatEvent,
    minimal: ChatEvent,
}

impl ServerExample {
    fn new(full: ChatEvent, minimal: ChatEvent) -> Self {
        Self { full, minimal }
    }

    /// A variant without optional field: `minimal` is `full`.
    fn same(event: ChatEvent) -> Self {
        Self {
            minimal: event.clone(),
            full: event,
        }
    }
}

/// `full` and `minimal` examples of every `ChatEvent` variant, in the order
/// of the enum definition.
fn server_examples() -> Vec<ServerExample> {
    let parent = || Some(s(PARENT_TOOL_USE_ID));

    vec![
        ServerExample::same(ChatEvent::UserMessage {
            content: s("Add a retry to the upload client."),
        }),
        ServerExample::same(ChatEvent::SystemHint {
            content: s("Context was compacted; the current task is restated below."),
        }),
        ServerExample::new(
            ChatEvent::AssistantText {
                content: s("I will start by reading the upload client."),
                parent_tool_use_id: parent(),
            },
            ChatEvent::AssistantText {
                content: s("I will start by reading the upload client."),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::Thinking {
                content: s("The retry must not replay a non-idempotent request."),
                parent_tool_use_id: parent(),
            },
            ChatEvent::Thinking {
                content: s("The retry must not replay a non-idempotent request."),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::ToolUse {
                id: s(TOOL_USE_ID),
                tool: s("Read"),
                input: json!({ "file_path": "src/upload/client.rs" }),
                parent_tool_use_id: parent(),
                category: Some(s("read")),
                canonical: Some(s("read_file")),
            },
            ChatEvent::ToolUse {
                id: s(TOOL_USE_ID),
                tool: s("Read"),
                input: json!({}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
        ),
        ServerExample::new(
            ChatEvent::ToolResult {
                id: s(TOOL_USE_ID),
                result: json!("pub struct UploadClient { /* ... */ }"),
                is_error: true,
                parent_tool_use_id: parent(),
            },
            ChatEvent::ToolResult {
                id: s(TOOL_USE_ID),
                result: json!(""),
                is_error: false,
                parent_tool_use_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::ToolUseInputResolved {
                id: s(TOOL_USE_ID),
                input: json!({ "file_path": "src/upload/client.rs" }),
                parent_tool_use_id: parent(),
            },
            ChatEvent::ToolUseInputResolved {
                id: s(TOOL_USE_ID),
                input: json!({}),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::ToolCancelled {
                id: s(TOOL_USE_ID),
                parent_tool_use_id: parent(),
            },
            ChatEvent::ToolCancelled {
                id: s(TOOL_USE_ID),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::PermissionRequest {
                id: s("perm_0001"),
                tool: s("Bash"),
                input: json!({ "command": "cargo fmt", "description": "Format the crate" }),
                parent_tool_use_id: parent(),
                category: Some(s("read")),
                canonical: Some(s("read_file")),
            },
            ChatEvent::PermissionRequest {
                id: s("perm_0001"),
                tool: s("Bash"),
                input: json!({}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
        ),
        ServerExample::new(
            ChatEvent::AskUserQuestion {
                id: s("ctrl_0001"),
                tool_call_id: s(TOOL_USE_ID),
                questions: json!([{
                    "question": "Which backoff should the retry use?",
                    "header": "Backoff",
                    "multiSelect": false,
                    "options": [
                        { "label": "Exponential", "description": "Doubles after each attempt" },
                        { "label": "Fixed", "description": "Same delay every time" }
                    ]
                }]),
                input: json!({
                    "questions": [{
                        "question": "Which backoff should the retry use?",
                        "header": "Backoff",
                        "multiSelect": false,
                        "options": [
                            { "label": "Exponential", "description": "Doubles after each attempt" },
                            { "label": "Fixed", "description": "Same delay every time" }
                        ]
                    }]
                }),
                parent_tool_use_id: parent(),
                synthetic: Some(true),
            },
            ChatEvent::AskUserQuestion {
                id: s("ctrl_0001"),
                tool_call_id: s(TOOL_USE_ID),
                questions: json!([]),
                input: json!({}),
                parent_tool_use_id: None,
                synthetic: None,
            },
        ),
        ServerExample::new(
            ChatEvent::Result {
                session_id: s(SESSION_ID),
                duration_ms: 48_210,
                cost_usd: Some(0.25),
                subtype: s("success"),
                is_error: false,
                num_turns: Some(7),
                result_text: Some(s("The retry is in place and covered by a test.")),
                cost: Some(json!({ "usd": 0.25, "basis": "reported" })),
                usage: Some(
                    json!({ "input_tokens": 1200, "output_tokens": 340, "cache_read_tokens": 800 }),
                ),
                model: Some(s("example-model-large")),
                stop_reason: Some(s("completed")),
            },
            ChatEvent::Result {
                session_id: s(SESSION_ID),
                duration_ms: 0,
                cost_usd: None,
                subtype: s(""),
                is_error: false,
                num_turns: None,
                result_text: None,
                cost: None,
                usage: None,
                model: None,
                stop_reason: None,
            },
        ),
        ServerExample::new(
            ChatEvent::StreamDelta {
                text: s("I will "),
                parent_tool_use_id: parent(),
            },
            ChatEvent::StreamDelta {
                text: s("I will "),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::same(ChatEvent::StreamingStatus { is_streaming: true }),
        ServerExample::new(
            ChatEvent::PendingQueue {
                messages: vec![PendingQueueEntry {
                    id: fixed_uuid(QUEUED_MESSAGE_ID),
                    content: s("Then update the changelog."),
                    attachments: vec![MessageAttachment {
                        id: fixed_uuid(DOCUMENT_ID),
                        filename: s("release-notes.pdf"),
                        mime_type: s("application/pdf"),
                        size_bytes: 182_044,
                    }],
                    queued_at: ts("2026-01-15T10:30:00Z"),
                    prioritized: true,
                }],
            },
            ChatEvent::PendingQueue { messages: vec![] },
        ),
        ServerExample::new(
            ChatEvent::Error {
                message: s("Tool execution failed: permission denied"),
                parent_tool_use_id: parent(),
            },
            ChatEvent::Error {
                message: s("Tool execution failed: permission denied"),
                parent_tool_use_id: None,
            },
        ),
        ServerExample::same(ChatEvent::PermissionDecision {
            id: s("perm_0001"),
            allow: true,
        }),
        ServerExample::new(
            ChatEvent::PermissionModeChanged {
                mode: s("acceptEdits"),
                policy_mode: Some(s("auto_edits")),
            },
            ChatEvent::PermissionModeChanged {
                mode: s("acceptEdits"),
                policy_mode: None,
            },
        ),
        ServerExample::same(ChatEvent::ModelChanged {
            model: s("example-model-large"),
        }),
        ServerExample::same(ChatEvent::CompactionStarted { trigger: s("auto") }),
        ServerExample::same(ChatEvent::CompactionRecovery {
            hint_tokens: 1_850,
            build_latency_ms: 240,
            recovery_success: true,
        }),
        ServerExample::new(
            ChatEvent::CompactBoundary {
                trigger: s("auto"),
                pre_tokens: Some(168_000),
            },
            ChatEvent::CompactBoundary {
                trigger: s("manual"),
                pre_tokens: None,
            },
        ),
        ServerExample::new(
            ChatEvent::SystemInit {
                cli_session_id: s("cli-session-0001"),
                model: Some(s("example-model-large")),
                tools: vec![s("Bash"), s("Read"), s("Edit")],
                mcp_servers: vec![json!({ "name": "project-orchestrator", "status": "connected" })],
                permission_mode: Some(s("default")),
                provider: Some(json!({ "id": "claude-code", "kind": "claude_code" })),
                capabilities: Some(json!({ "interactive_permissions": true, "images": false })),
                tool_policy: Some(
                    json!({ "mode": "ask", "native_mode": "default", "allow": [], "deny": [] }),
                ),
                policy_mode: Some(s("ask")),
                engine: Some(s("agent")),
                degraded_features: Some(vec![s("hooks"), s("message_queue")]),
            },
            ChatEvent::SystemInit {
                cli_session_id: s("cli-session-0001"),
                model: None,
                tools: vec![],
                mcp_servers: vec![],
                permission_mode: None,
                provider: None,
                capabilities: None,
                tool_policy: None,
                policy_mode: None,
                engine: None,
                degraded_features: None,
            },
        ),
        ServerExample::new(
            ChatEvent::Workflow {
                subtype: s("task_progress"),
                data: json!({
                    "uuid": "3c2b1a09-8f7e-4d6c-8b5a-4c3d2e1f0a9b",
                    "task_id": "wf_task_0001",
                    "workflow_name": "review",
                    "status": "running",
                    "workflow_progress": [
                        { "agent": "reviewer-1", "status": "running" },
                        { "agent": "reviewer-2", "status": "completed" }
                    ]
                }),
            },
            ChatEvent::Workflow {
                subtype: s("task_started"),
                data: json!({}),
            },
        ),
        ServerExample::same(ChatEvent::AutoContinue {
            session_id: s(SESSION_ID),
            delay_ms: 3_000,
        }),
        ServerExample::same(ChatEvent::AutoContinueStateChanged {
            session_id: s(SESSION_ID),
            enabled: true,
        }),
        ServerExample::same(ChatEvent::Retrying {
            attempt: 1,
            max_attempts: 3,
            delay_ms: 2_000,
            error_message: s("API Error: 529 Overloaded"),
        }),
        ServerExample::new(
            ChatEvent::BackgroundOutput {
                source: s("Monitor"),
                content: s("ERROR upload failed: connection reset"),
                received_at: ts("2026-01-15T10:31:12Z"),
                correlation_id: Some(s(TOOL_USE_ID)),
            },
            ChatEvent::BackgroundOutput {
                source: s("system"),
                content: s("ERROR upload failed: connection reset"),
                received_at: ts("2026-01-15T10:31:12Z"),
                correlation_id: None,
            },
        ),
        ServerExample::new(
            ChatEvent::ToolsCancelled {
                cli_pid: Some(48_211),
                killed_count: 2,
                requested_by: s("user"),
            },
            ChatEvent::ToolsCancelled {
                cli_pid: None,
                killed_count: 0,
                requested_by: s("user"),
            },
        ),
        ServerExample::same(ChatEvent::SessionError {
            reason: s("subprocess_exited"),
            message: s("The agent process stopped. Send a message to restart the session."),
            received_at: ts("2026-01-15T10:32:00Z"),
        }),
        ServerExample::new(
            ChatEvent::ActiveTasksUpdate {
                tasks: vec![
                    BackgroundTaskInfo {
                        id: s(TOOL_USE_ID),
                        kind: BackgroundTaskKind::Monitor,
                        description: s("watch deploy.log for errors"),
                        started_at: ts("2026-01-15T10:30:00Z"),
                        pid: Some(48_300),
                        parent_tool_use_id: Some(s(TOOL_USE_ID)),
                        last_seen_at: ts("2026-01-15T10:31:12Z"),
                        pending_removal_at: None,
                    },
                    BackgroundTaskInfo {
                        id: s("recovered-48301"),
                        kind: BackgroundTaskKind::BashBackground,
                        description: s("cargo watch"),
                        started_at: ts("2026-01-15T10:30:30Z"),
                        pid: None,
                        parent_tool_use_id: None,
                        last_seen_at: ts("2026-01-15T10:30:30Z"),
                        pending_removal_at: None,
                    },
                ],
            },
            ChatEvent::ActiveTasksUpdate { tasks: vec![] },
        ),
        ServerExample::same(ChatEvent::SecretRequest {
            id: s("secret_req_0001"),
            name: s("REGISTRY_TOKEN"),
            reason: s("Needed to publish the package."),
            exists: false,
        }),
        ServerExample::same(ChatEvent::SecretRequestResolved {
            id: s("secret_req_0001"),
            outcome: s("provided"),
        }),
        ServerExample::new(
            ChatEvent::SessionClosed {
                session_id: s(SESSION_ID),
                reason: Some(s("closed")),
            },
            ChatEvent::SessionClosed {
                session_id: s(SESSION_ID),
                reason: None,
            },
        ),
    ]
}

/// The two example frames of one client message, as JSON (the type is only
/// `Deserialize`).
struct ClientExample {
    tag: &'static str,
    full: Value,
    minimal: Value,
}

/// `full` and `minimal` examples of every `WsChatClientMessage` variant, in
/// the order of the enum definition.
///
/// `queue_op` flattens `chat::pending_queue::QueueOp` (tagged by `op`): its
/// `full` frame is the widest operation (`edit`), its `minimal` frame the
/// narrowest (`snapshot`); `remove`, `prioritize` and `send_now` take `op`
/// and `id`.
fn client_examples() -> Vec<ClientExample> {
    vec![
        ClientExample {
            tag: "user_message",
            full: json!({
                "type": "user_message",
                "content": "Add a retry to the upload client.",
                "attachments": [DOCUMENT_ID],
                "queue": true
            }),
            minimal: json!({
                "type": "user_message",
                "content": "Add a retry to the upload client."
            }),
        },
        ClientExample {
            tag: "queue_op",
            full: json!({
                "type": "queue_op",
                "op": "edit",
                "id": QUEUED_MESSAGE_ID,
                "content": "Then update the changelog and the README."
            }),
            minimal: json!({ "type": "queue_op", "op": "snapshot" }),
        },
        ClientExample {
            tag: "interrupt",
            full: json!({ "type": "interrupt" }),
            minimal: json!({ "type": "interrupt" }),
        },
        ClientExample {
            tag: "permission_response",
            full: json!({ "type": "permission_response", "id": "perm_0001", "allow": true }),
            minimal: json!({ "type": "permission_response" }),
        },
        ClientExample {
            tag: "input_response",
            full: json!({
                "type": "input_response",
                "id": TOOL_USE_ID,
                "content": "Exponential"
            }),
            minimal: json!({ "type": "input_response", "content": "Exponential" }),
        },
        ClientExample {
            tag: "set_permission_mode",
            full: json!({ "type": "set_permission_mode", "mode": "acceptEdits" }),
            minimal: json!({ "type": "set_permission_mode", "mode": "acceptEdits" }),
        },
        ClientExample {
            tag: "set_model",
            full: json!({ "type": "set_model", "model": "example-model-large" }),
            minimal: json!({ "type": "set_model", "model": "example-model-large" }),
        },
        ClientExample {
            tag: "set_auto_continue",
            full: json!({ "type": "set_auto_continue", "enabled": true }),
            minimal: json!({ "type": "set_auto_continue", "enabled": true }),
        },
        ClientExample {
            tag: "cancel_tools",
            full: json!({ "type": "cancel_tools" }),
            minimal: json!({ "type": "cancel_tools" }),
        },
    ]
}

/// A frame the WS handler builds by hand (not a `ChatEvent`).
struct ControlFrame {
    tag: &'static str,
    /// Function that sends it, as `module::function`.
    source: &'static str,
    full: Value,
    minimal: Value,
}

impl ControlFrame {
    fn same(tag: &'static str, source: &'static str, frame: Value) -> Self {
        Self {
            tag,
            source,
            minimal: frame.clone(),
            full: frame,
        }
    }
}

/// Static copies of the control frames sent to the client.
fn control_frames() -> Vec<ControlFrame> {
    vec![
        // api::ws_auth::send_auth_ok (called by wait_ready_then_auth_ok)
        ControlFrame::same(
            "auth_ok",
            "api::ws_auth::send_auth_ok",
            json!({
                "type": "auth_ok",
                "user": {
                    "id": "7d6c5b4a-3f2e-4d1c-8b0a-9f8e7d6c5b4a",
                    "email": "dev@example.com",
                    "name": "Example Developer"
                }
            }),
        ),
        // api::ws_chat_handler::handle_ws_chat_loop — mid-stream join snapshot
        ControlFrame::same(
            "partial_text",
            "api::ws_chat_handler::handle_ws_chat_loop",
            json!({
                "type": "partial_text",
                "content": "I will start by reading the upload",
                "seq": 0,
                "replaying": true
            }),
        ),
        // api::ws_chat_handler::handle_ws_chat_loop — end of replay + snapshot
        ControlFrame::same(
            "replay_complete",
            "api::ws_chat_handler::handle_ws_chat_loop",
            json!({ "type": "replay_complete" }),
        ),
        // api::ws_chat_handler::handle_ws_chat_loop — broadcast receiver lagged
        ControlFrame::same(
            "events_lagged",
            "api::ws_chat_handler::handle_ws_chat_loop",
            json!({ "type": "events_lagged", "skipped": 12 }),
        ),
        // api::ws_chat_handler::handle_ws_chat_loop — broadcast closed (idle cleanup)
        ControlFrame::same(
            "session_dormant",
            "api::ws_chat_handler::handle_ws_chat_loop",
            json!({
                "type": "session_dormant",
                "message": "CLI session cleaned up (idle timeout). Will resume on next message."
            }),
        ),
    ]
}

/// Fields the WS handler ADDS to a serialized `ChatEvent` before sending it.
///
/// api::ws_chat_handler::handle_ws_chat_loop: live events get `seq: 0`
/// (`send_chat_event!`), replayed events get their stored `seq` and
/// `replaying: true`. The frames the handler builds itself with a `ChatEvent`
/// tag (`error`, `streaming_status`, `pending_queue`,
/// `permission_mode_changed`, `model_changed`) carry neither, hence both
/// fields are optional.
fn event_envelope() -> (Value, Value) {
    (json!({ "seq": 42, "replaying": true }), json!({}))
}

// ============================================================================
// Contract derivation
// ============================================================================

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Per-field contract of one frame, from its two examples.
///
/// - `type`: JSON type of the field in `full`;
/// - `required`: the field is present in `minimal`;
/// - `nullable`: present (only when true) if the field is `null` in either
///   example.
fn derive_fields(tag: &str, full: &Value, minimal: &Value) -> Value {
    let full_obj = full
        .as_object()
        .unwrap_or_else(|| panic!("`{tag}`: the full example is not a JSON object"));
    let minimal_obj = minimal
        .as_object()
        .unwrap_or_else(|| panic!("`{tag}`: the minimal example is not a JSON object"));

    for name in minimal_obj.keys() {
        assert!(
            full_obj.contains_key(name),
            "`{tag}`: field `{name}` is in the minimal example but not in the full one"
        );
    }

    let mut fields: BTreeMap<String, Value> = BTreeMap::new();
    for (name, value) in full_obj {
        let in_minimal = minimal_obj.get(name);
        if let Some(min_value) = in_minimal {
            assert!(
                value.is_null() || min_value.is_null() || json_type(value) == json_type(min_value),
                "`{tag}`: field `{name}` is `{}` in the full example and `{}` in the minimal one",
                json_type(value),
                json_type(min_value)
            );
        }

        let mut spec = serde_json::Map::new();
        spec.insert("type".to_string(), json!(json_type(value)));
        spec.insert("required".to_string(), json!(in_minimal.is_some()));
        if value.is_null() || in_minimal.is_some_and(Value::is_null) {
            spec.insert("nullable".to_string(), json!(true));
        }
        fields.insert(name.clone(), Value::Object(spec));
    }
    json!(fields)
}

fn contract_entry(tag: &str, full: Value, minimal: Value) -> Value {
    json!({
        "fields": derive_fields(tag, &full, &minimal),
        "examples": { "full": full, "minimal": minimal }
    })
}

fn server_events_doc() -> Value {
    let mut events: BTreeMap<String, Value> = BTreeMap::new();
    for example in server_examples() {
        let tag = variant_tag(&example.full);
        let full = serde_json::to_value(&example.full).expect("ChatEvent serializes");
        let minimal = serde_json::to_value(&example.minimal).expect("ChatEvent serializes");
        let previous = events.insert(tag.to_string(), contract_entry(tag, full, minimal));
        assert!(previous.is_none(), "two sets of examples for `{tag}`");
    }
    json!({ "contract_version": CONTRACT_VERSION, "events": events })
}

fn client_messages_doc() -> Value {
    let mut messages: BTreeMap<String, Value> = BTreeMap::new();
    for example in client_examples() {
        let entry = contract_entry(example.tag, example.full, example.minimal);
        let previous = messages.insert(example.tag.to_string(), entry);
        assert!(
            previous.is_none(),
            "two sets of examples for `{}`",
            example.tag
        );
    }
    json!({ "contract_version": CONTRACT_VERSION, "messages": messages })
}

fn control_frames_doc() -> Value {
    let mut frames: BTreeMap<String, Value> = BTreeMap::new();
    for frame in control_frames() {
        let mut entry = contract_entry(frame.tag, frame.full, frame.minimal);
        entry["source"] = json!(frame.source);
        let previous = frames.insert(frame.tag.to_string(), entry);
        assert!(
            previous.is_none(),
            "two sets of examples for `{}`",
            frame.tag
        );
    }

    let (envelope_full, envelope_minimal) = event_envelope();
    let mut envelope = contract_entry("event_envelope", envelope_full, envelope_minimal);
    envelope["source"] = json!("api::ws_chat_handler::handle_ws_chat_loop");

    json!({
        "contract_version": CONTRACT_VERSION,
        "frames": frames,
        "event_envelope": envelope
    })
}

/// Rebuild every object with its keys in ascending order, so the output does
/// not depend on how `serde_json::Map` is configured (`preserve_order`).
fn sort_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> =
                map.into_iter().map(|(k, v)| (k, sort_keys(v))).collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_keys).collect()),
        other => other,
    }
}

/// File content: sorted keys, pretty-printed, one trailing newline.
fn render(doc: Value) -> String {
    let mut text = serde_json::to_string_pretty(&sort_keys(doc)).expect("JSON value serializes");
    text.push('\n');
    text
}

/// Every fixture file as `(name, content)`: the three JSON files in name
/// order, then `SHA256SUMS` (same format as `shasum -a 256`).
fn generated_files() -> Vec<(&'static str, String)> {
    let mut files = vec![
        (CLIENT_MESSAGES_FILE, render(client_messages_doc())),
        (CONTROL_FRAMES_FILE, render(control_frames_doc())),
        (SERVER_EVENTS_FILE, render(server_events_doc())),
    ];

    let mut sums = String::new();
    for (name, content) in &files {
        let digest = hex::encode(Sha256::digest(content.as_bytes()));
        sums.push_str(&format!("{digest}  {name}\n"));
    }
    files.push((CHECKSUMS_FILE, sums));
    files
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn sorted(tags: &[&str]) -> Vec<String> {
    let mut list: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    list.sort();
    list
}

// ============================================================================
// Tests
// ============================================================================

/// Regenerates the fixtures in memory and compares them byte for byte to the
/// committed files; with `UPDATE_CHAT_CONTRACT=1` it writes them instead.
#[test]
fn wire_contract_matches_committed_fixtures() {
    let dir = fixture_dir();
    let files = generated_files();

    if std::env::var(UPDATE_ENV).as_deref() == Ok("1") {
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        for (name, content) in &files {
            let path = dir.join(name);
            std::fs::write(&path, content)
                .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        }
        return;
    }

    let mut stale: Vec<String> = Vec::new();
    for (name, expected) in &files {
        let path = dir.join(name);
        match std::fs::read_to_string(&path) {
            Ok(committed) if committed == *expected => {}
            Ok(_) => stale.push(format!("{name} (differs)")),
            Err(e) => stale.push(format!("{name} (unreadable: {e})")),
        }
    }

    assert!(
        stale.is_empty(),
        "the chat wire contract in {} is out of date: {}.\n\
         The wire changed (or the fixtures were edited by hand). Regenerate them and commit \
         the result in the same commit:\n    {UPDATE_COMMAND}",
        dir.display(),
        stale.join(", ")
    );
}

#[test]
fn examples_cover_every_chat_event_variant() {
    let examples = server_examples();
    let mut tags: Vec<String> = Vec::new();
    for example in &examples {
        let tag = variant_tag(&example.full);
        assert_eq!(
            tag,
            variant_tag(&example.minimal),
            "the full and minimal examples of `{tag}` are not the same variant"
        );
        assert_eq!(
            tag,
            example.full.event_type(),
            "variant_tag and ChatEvent::event_type disagree"
        );
        let wire = serde_json::to_value(&example.full).expect("ChatEvent serializes");
        assert_eq!(
            wire["type"],
            json!(tag),
            "serde tag differs from variant_tag"
        );
        tags.push(tag.to_string());
    }
    tags.sort();

    let mut unique = tags.clone();
    unique.dedup();
    assert_eq!(tags, unique, "a variant has more than one set of examples");
    assert_eq!(
        tags,
        sorted(&EXPECTED_SERVER_TAGS),
        "the examples and EXPECTED_SERVER_TAGS do not list the same variants"
    );
}

#[test]
fn every_example_roundtrips() {
    for example in server_examples() {
        for event in [&example.full, &example.minimal] {
            let tag = variant_tag(event);
            let wire = serde_json::to_value(event).expect("ChatEvent serializes");
            let back: ChatEvent = serde_json::from_value(wire.clone())
                .unwrap_or_else(|e| panic!("`{tag}` does not deserialize: {e}"));
            assert_eq!(
                variant_tag(&back),
                tag,
                "`{tag}` came back as another variant"
            );
            let again = serde_json::to_value(&back).expect("ChatEvent serializes");
            assert_eq!(wire, again, "`{tag}` does not round-trip unchanged");
        }
    }
}

#[test]
fn client_examples_parse_as_their_variant() {
    let examples = client_examples();
    let mut tags: Vec<&str> = Vec::new();
    for example in &examples {
        for (label, frame) in [("full", &example.full), ("minimal", &example.minimal)] {
            assert_eq!(
                frame["type"],
                json!(example.tag),
                "`{}` ({label}): wrong `type`",
                example.tag
            );
            let parsed: WsChatClientMessage = serde_json::from_value(frame.clone())
                .unwrap_or_else(|e| panic!("`{}` ({label}) does not parse: {e}", example.tag));
            assert_eq!(
                client_tag(&parsed),
                example.tag,
                "`{}` ({label}) parsed as another variant",
                example.tag
            );
        }
        tags.push(example.tag);
    }
    assert_eq!(
        sorted(&tags),
        sorted(&EXPECTED_CLIENT_TAGS),
        "the examples and EXPECTED_CLIENT_TAGS do not list the same variants"
    );
}

/// `required` is derived from the hand-written minimal frame, so check it
/// against the real type: dropping any field of a minimal frame must make it
/// unparseable.
#[test]
fn client_minimal_examples_hold_only_required_fields() {
    for example in client_examples() {
        let minimal = example
            .minimal
            .as_object()
            .expect("minimal example is an object");
        for name in minimal.keys() {
            let mut without = minimal.clone();
            without.remove(name);
            let parsed = serde_json::from_value::<WsChatClientMessage>(Value::Object(without));
            assert!(
                parsed.is_err(),
                "`{}`: field `{name}` is in the minimal example but the message parses without it",
                example.tag
            );
        }
    }
}

#[test]
fn control_frames_are_tagged_and_do_not_shadow_chat_events() {
    let mut tags: Vec<&str> = Vec::new();
    for frame in control_frames() {
        assert_eq!(frame.full["type"], json!(frame.tag));
        assert_eq!(frame.minimal["type"], json!(frame.tag));
        assert!(
            !EXPECTED_SERVER_TAGS.contains(&frame.tag),
            "`{}` is a ChatEvent tag, not a control frame",
            frame.tag
        );
        assert!(!frame.source.is_empty());
        tags.push(frame.tag);
    }
    let mut unique = sorted(&tags);
    unique.dedup();
    assert_eq!(unique.len(), tags.len(), "duplicate control frame tag");
}

#[test]
fn derived_fields_mark_optional_and_nullable() {
    let doc = server_events_doc();

    // `skip_serializing_if` → not required.
    let assistant = &doc["events"]["assistant_text"]["fields"];
    assert_eq!(
        assistant["content"],
        json!({ "type": "string", "required": true })
    );
    assert_eq!(
        assistant["parent_tool_use_id"],
        json!({ "type": "string", "required": false })
    );

    // `Option` without `skip_serializing_if` → always present, may be null.
    let result = &doc["events"]["result"]["fields"];
    assert_eq!(
        result["cost_usd"],
        json!({ "type": "number", "required": true, "nullable": true })
    );
    assert_eq!(
        result["num_turns"],
        json!({ "type": "number", "required": false })
    );

    // Empty `Vec` skipped → not required.
    let init = &doc["events"]["system_init"]["fields"];
    assert_eq!(init["tools"], json!({ "type": "array", "required": false }));
}

#[test]
fn rendering_is_stable_and_sorted() {
    let text = render(json!({ "b": { "z": 1, "a": [ { "y": 2, "x": 3 } ] }, "a": null }));
    assert_eq!(
        text,
        "{\n  \"a\": null,\n  \"b\": {\n    \"a\": [\n      {\n        \"x\": 3,\n        \"y\": 2\n      }\n    ],\n    \"z\": 1\n  }\n}\n"
    );
    assert_eq!(generated_files(), generated_files());
}
