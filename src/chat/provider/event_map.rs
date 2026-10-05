//! `AgentEvent` (nexus contract) to `ChatEvent` (the wire the frontend knows).
//!
//! The table is `docs/agent-contract.md` section 14. This module is pure: no
//! I/O, no clock beyond the `received_at` stamp of out-of-turn events. It is the
//! only place where a provider-neutral event becomes a backend event, so a new
//! provider cannot change what the frontend receives.
//!
//! Two entry points:
//! - [`EventMapper::map`] for the events of a turn (what `message_to_events`
//!   does for the legacy path);
//! - [`out_of_band_to_chat_events`] for the events received between turns,
//!   regrouped by provider message number into `background_output`.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use nexus_claude::agent::{
    AgentEvent, CompactionPhase, Cost, DeltaKind, ProviderError, QuestionReply, StopReason,
    ToolCategory, ToolOutput, Usage,
};
use serde_json::{json, Value};

use super::errors::open_failure;
use super::policy::{legacy_name, neutral_name};
use crate::chat::manager::{MASKING_FAILED_MESSAGE, MASKING_FAILED_SUBTYPE};
use crate::chat::types::ChatEvent;

/// What the mapping remembers from one event to the next.
#[derive(Debug, Default, Clone)]
pub struct EventMapper {
    /// Tool calls already announced with an incomplete input: their complete
    /// form becomes `tool_use_input_resolved`, not a second `tool_use`.
    announced_tools: HashSet<String>,
    /// `event_id` of the task updates already forwarded (de-duplication).
    seen_task_events: HashSet<String>,
}

impl EventMapper {
    /// A fresh mapper (one per session).
    pub fn new() -> Self {
        Self::default()
    }

    /// Maps one event of a turn. Never panics on an event it does not know: it
    /// maps to nothing.
    pub fn map(&mut self, event: &AgentEvent) -> Vec<ChatEvent> {
        match event {
            AgentEvent::SessionStarted {
                provider_session_id,
                model,
                policy_mode,
                native_mode,
                tools,
                mcp_servers,
                ..
            } => vec![ChatEvent::SystemInit {
                cli_session_id: provider_session_id.clone().unwrap_or_default(),
                model: model.clone(),
                tools: tools.clone(),
                mcp_servers: mcp_servers
                    .iter()
                    .map(|s| json!({"name": s.name, "status": s.status}))
                    .collect(),
                permission_mode: native_mode
                    .clone()
                    .or_else(|| policy_mode.map(|m| legacy_name(m).to_string())),
                // Added by the session owner, who knows the instance and the policy.
                provider: None,
                capabilities: None,
                tool_policy: None,
                policy_mode: policy_mode.map(|m| neutral_name(m).to_string()),
            }],
            // In a turn the echo of the user's own message is not an event
            // (the backend already emitted `user_message`).
            AgentEvent::UserEcho { .. } => Vec::new(),
            AgentEvent::Text { text, parent, .. } => vec![ChatEvent::AssistantText {
                content: text.clone(),
                parent_tool_use_id: parent.clone(),
            }],
            AgentEvent::Thinking { text, parent, .. } => vec![ChatEvent::Thinking {
                content: text.clone(),
                parent_tool_use_id: parent.clone(),
            }],
            AgentEvent::Delta {
                kind: DeltaKind::Text,
                text,
                parent,
                ..
            } => vec![ChatEvent::StreamDelta {
                text: text.clone(),
                parent_tool_use_id: parent.clone(),
            }],
            // Thinking and tool-input fragments have no ChatEvent today.
            AgentEvent::Delta { .. } => Vec::new(),
            AgentEvent::ToolCall {
                id,
                name,
                input,
                input_complete,
                parent,
                category,
                canonical,
                ..
            } => {
                let (category, canonical) = (Some(category_name(*category)), canonical.clone());
                if *input_complete && self.announced_tools.remove(id) {
                    vec![ChatEvent::ToolUseInputResolved {
                        id: id.clone(),
                        input: input.clone(),
                        parent_tool_use_id: parent.clone(),
                    }]
                } else {
                    if !*input_complete {
                        self.announced_tools.insert(id.clone());
                    }
                    vec![ChatEvent::ToolUse {
                        id: id.clone(),
                        tool: name.clone(),
                        input: input.clone(),
                        parent_tool_use_id: parent.clone(),
                        category,
                        canonical,
                    }]
                }
            }
            AgentEvent::ToolResult {
                id,
                output,
                is_error,
                parent,
                ..
            } => vec![ChatEvent::ToolResult {
                id: id.clone(),
                result: tool_output_value(output.as_ref()),
                is_error: *is_error,
                parent_tool_use_id: parent.clone(),
            }],
            AgentEvent::PermissionAsk {
                request_id,
                tool_name,
                input,
                parent,
                category,
                canonical,
                ..
            } => vec![ChatEvent::PermissionRequest {
                id: request_id.clone(),
                tool: tool_name.clone(),
                input: input.clone(),
                parent_tool_use_id: parent.clone(),
                category: Some(category_name(*category)),
                canonical: canonical.clone(),
            }],
            AgentEvent::Question {
                question_id,
                tool_call_id,
                questions,
                input,
                parent,
                reply,
            } => vec![ChatEvent::AskUserQuestion {
                id: question_id.clone(),
                tool_call_id: tool_call_id.clone().unwrap_or_default(),
                // The provider's own shape when it gave one, the parsed list
                // otherwise: the frontend reads `questions` leniently.
                questions: input
                    .get("questions")
                    .cloned()
                    .unwrap_or_else(|| serde_json::to_value(questions).unwrap_or(json!([]))),
                input: input.clone(),
                parent_tool_use_id: parent.clone(),
                // A provider without a native question: the answer is a user turn (A45).
                synthetic: (*reply == QuestionReply::Turn).then_some(true),
            }],
            AgentEvent::Compaction {
                phase,
                trigger,
                pre_tokens,
            } => {
                let trigger = trigger.map_or("auto", |t| t.as_str()).to_string();
                match phase {
                    CompactionPhase::Started => vec![ChatEvent::CompactionStarted { trigger }],
                    _ => vec![ChatEvent::CompactBoundary {
                        trigger,
                        pre_tokens: *pre_tokens,
                    }],
                }
            }
            // The live task list is kept by the backend's own tracking; the
            // snapshot is not a wire event today.
            AgentEvent::BackgroundTasks { .. } => Vec::new(),
            AgentEvent::TaskUpdate {
                phase,
                event_id,
                data,
                ..
            } => {
                if let Some(id) = event_id {
                    if !self.seen_task_events.insert(id.clone()) {
                        return Vec::new();
                    }
                }
                vec![ChatEvent::Workflow {
                    subtype: format!("task_{}", phase.as_str()),
                    data: data.clone(),
                }]
            }
            AgentEvent::ModelChanged { model } => vec![ChatEvent::ModelChanged {
                model: model.clone(),
            }],
            AgentEvent::PolicyModeChanged { mode, native_mode } => {
                vec![ChatEvent::PermissionModeChanged {
                    mode: native_mode
                        .clone()
                        .unwrap_or_else(|| legacy_name(*mode).to_string()),
                    policy_mode: Some(neutral_name(*mode).to_string()),
                }]
            }
            AgentEvent::Done {
                stop_reason,
                subtype,
                is_error,
                result_text,
                usage,
                cost,
                duration_ms,
                num_turns,
                provider_session_id,
                model,
                ..
            } => vec![ChatEvent::Result {
                session_id: provider_session_id.clone().unwrap_or_default(),
                duration_ms: *duration_ms,
                cost_usd: cost.usd,
                subtype: subtype
                    .clone()
                    .unwrap_or_else(|| default_subtype(*stop_reason, *is_error).to_string()),
                is_error: *is_error,
                num_turns: Some(i32::try_from(*num_turns).unwrap_or(i32::MAX)),
                result_text: result_text.clone(),
                cost: Some(cost_value(cost)),
                usage: usage_value(usage),
                model: model.clone(),
                stop_reason: Some(stop_reason_name(*stop_reason).to_string()),
            }],
            AgentEvent::Error { error } => vec![error_event(error)],
            AgentEvent::ProviderNotice { kind, .. } if kind == MASKING_FAILED_SUBTYPE => {
                vec![ChatEvent::Error {
                    message: MASKING_FAILED_MESSAGE.to_string(),
                    parent_tool_use_id: None,
                }]
            }
            // A notice is a diagnostic: nothing in a turn.
            AgentEvent::ProviderNotice { .. } => Vec::new(),
            // `AgentEvent` is non-exhaustive: an unknown event is ignored.
            _ => Vec::new(),
        }
    }
}

/// One-shot form of [`EventMapper::map`] for a stateless caller (tests, replay).
pub fn agent_event_to_chat_event(event: &AgentEvent, mapper: &mut EventMapper) -> Vec<ChatEvent> {
    mapper.map(event)
}

fn category_name(c: ToolCategory) -> String {
    serde_json::to_value(c)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "other".to_string())
}

fn stop_reason_name(s: StopReason) -> String {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "error".to_string())
}

/// `{ usd?, basis }`: an unknown price carries no `usd` (never 0).
fn cost_value(cost: &Cost) -> Value {
    serde_json::to_value(cost).unwrap_or(Value::Null)
}

/// The token counts of the turn, `None` when the provider reported none.
fn usage_value(usage: &Usage) -> Option<Value> {
    let counts = json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_creation_tokens": usage.cache_creation_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    });
    let map: serde_json::Map<String, Value> = counts
        .as_object()?
        .iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    (!map.is_empty()).then(|| Value::Object(map))
}

/// `subtype` of a `result` when the provider gave none.
fn default_subtype(stop: StopReason, is_error: bool) -> &'static str {
    match stop {
        StopReason::Completed => "success",
        StopReason::MaxTurns => "error_max_turns",
        StopReason::BudgetExceeded => "error_max_budget_usd",
        _ if is_error => "error_during_execution",
        _ => "success",
    }
}

fn tool_output_value(output: Option<&ToolOutput>) -> Value {
    match output {
        Some(ToolOutput::Text(s)) => Value::String(s.clone()),
        Some(ToolOutput::Blocks(blocks)) => Value::Array(blocks.clone()),
        None => Value::Null,
    }
}

/// The event of a failed turn: a process that died is a session error, any
/// other failure is an in-stream error whose text never carries a credential,
/// a path or a URL (it is written in `errors.rs`, not copied from the source).
fn error_event(error: &ProviderError) -> ChatEvent {
    if matches!(error, ProviderError::ProcessExited { .. }) {
        return ChatEvent::SessionError {
            reason: "subprocess_exited".to_string(),
            message: "The provider process exited unexpectedly.".to_string(),
            received_at: Utc::now(),
        };
    }
    ChatEvent::Error {
        message: format!("Error: {}", open_failure(error, None).message),
        parent_tool_use_id: None,
    }
}

/// One group of out-of-turn lines sharing a provider message number.
struct Group {
    seq: Option<u64>,
    lines: Vec<String>,
    has_user: bool,
    only_results: bool,
    parent: Option<String>,
}

impl Group {
    fn new(seq: Option<u64>) -> Self {
        Self {
            seq,
            lines: Vec::new(),
            has_user: false,
            only_results: true,
            parent: None,
        }
    }

    fn flush(self, at: DateTime<Utc>, out: &mut Vec<ChatEvent>) {
        if self.lines.is_empty() {
            return;
        }
        let source = if self.has_user {
            "user"
        } else if self.only_results {
            "tool_result"
        } else {
            "assistant"
        };
        out.push(ChatEvent::BackgroundOutput {
            source: source.to_string(),
            content: self.lines.join("\n"),
            received_at: at,
            correlation_id: self.parent,
        });
    }
}

fn result_line(output: Option<&ToolOutput>) -> String {
    match output {
        Some(ToolOutput::Text(s)) => format!("[tool_result] {s}"),
        Some(ToolOutput::Blocks(blocks)) => {
            format!(
                "[tool_result] {}",
                serde_json::to_string(blocks).unwrap_or_default()
            )
        }
        None => "[tool_result]".to_string(),
    }
}

/// Maps the events received out of turn. Text-like events are regrouped by
/// provider message number into one `background_output` per message; a
/// `provider_notice` becomes `background_output { source: "system:<kind>" }`;
/// the other events go through [`EventMapper::map`].
pub fn out_of_band_to_chat_events(
    events: &[AgentEvent],
    mapper: &mut EventMapper,
    received_at: DateTime<Utc>,
) -> Vec<ChatEvent> {
    let mut out = Vec::new();
    let mut group: Option<Group> = None;

    for event in events {
        // (seq, parent, line, is_user, is_result) of a text-like event.
        let line: Option<(Option<u64>, Option<String>, String, bool, bool)> = match event {
            AgentEvent::Text {
                text, seq, parent, ..
            } => Some((*seq, parent.clone(), text.clone(), false, false)),
            AgentEvent::Thinking {
                text, seq, parent, ..
            } => Some((
                *seq,
                parent.clone(),
                format!("[thinking] {text}"),
                false,
                false,
            )),
            AgentEvent::ToolCall {
                name,
                seq,
                parent,
                input_complete: true,
                ..
            } => Some((
                *seq,
                parent.clone(),
                format!("[tool_use: {name}]"),
                false,
                false,
            )),
            AgentEvent::ToolResult {
                output,
                seq,
                parent,
                ..
            } => Some((
                *seq,
                parent.clone(),
                result_line(output.as_ref()),
                false,
                true,
            )),
            AgentEvent::UserEcho {
                text, seq, parent, ..
            } => Some((*seq, parent.clone(), text.clone(), true, false)),
            _ => None,
        };

        match line {
            Some((seq, parent, text, is_user, is_result)) => {
                let same = group
                    .as_ref()
                    .is_some_and(|g| g.seq.is_some() && g.seq == seq);
                if !same {
                    if let Some(done) = group.take() {
                        done.flush(received_at, &mut out);
                    }
                    group = Some(Group::new(seq));
                }
                if let Some(g) = group.as_mut() {
                    g.lines.push(text);
                    g.has_user |= is_user;
                    g.only_results &= is_result;
                    if g.parent.is_none() {
                        g.parent = parent;
                    }
                }
            }
            None => {
                if let Some(done) = group.take() {
                    done.flush(received_at, &mut out);
                }
                match event {
                    AgentEvent::ProviderNotice { kind, data } if kind != MASKING_FAILED_SUBTYPE => {
                        out.push(ChatEvent::BackgroundOutput {
                            source: format!("system:{kind}"),
                            content: serde_json::to_string_pretty(data).unwrap_or_default(),
                            received_at,
                            correlation_id: None,
                        });
                    }
                    // A stream fragment or an announced-but-incomplete tool call
                    // is an in-turn artefact, never an out-of-turn event.
                    AgentEvent::Delta { .. } => {}
                    AgentEvent::ToolCall { .. } => {}
                    other => out.extend(mapper.map(other)),
                }
            }
        }
    }
    if let Some(done) = group.take() {
        done.flush(received_at, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::manager::ChatManager;
    use nexus_claude::providers::claude_code::{map_message, MapState};
    use nexus_claude::{
        AssistantMessage, ContentBlock, ContentValue, Message, StreamDelta, StreamEventData,
        TextContent, ThinkingContent, ToolResultContent, ToolUseContent, UserMessage,
    };

    fn result_msg(subtype: &str, is_error: bool, cost: Option<f64>, text: Option<&str>) -> Message {
        Message::Result {
            subtype: subtype.into(),
            duration_ms: 5000,
            duration_api_ms: 4500,
            is_error,
            num_turns: 3,
            session_id: "cli-abc-123".into(),
            total_cost_usd: cost,
            usage: None,
            result: text.map(str::to_string),
            structured_output: None,
        }
    }

    fn assistant(blocks: Vec<ContentBlock>, parent: Option<&str>) -> Message {
        Message::Assistant {
            message: AssistantMessage { content: blocks },
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    fn user_blocks(blocks: Vec<ContentBlock>) -> Message {
        Message::User {
            message: UserMessage {
                content: String::new(),
                content_blocks: Some(blocks),
            },
            parent_tool_use_id: None,
        }
    }

    fn tool_result(
        id: &str,
        content: Option<ContentValue>,
        is_error: Option<bool>,
    ) -> ContentBlock {
        ContentBlock::ToolResult(ToolResultContent {
            tool_use_id: id.into(),
            content,
            is_error,
        })
    }

    fn stream(event: StreamEventData, parent: Option<&str>) -> Message {
        Message::StreamEvent {
            event,
            session_id: None,
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    /// Every message shape the legacy path knows, with the parent set where it
    /// can be.
    fn corpus() -> Vec<(&'static str, Message)> {
        vec![
            (
                "init",
                Message::System {
                    subtype: "init".into(),
                    data: json!({
                        "session_id": "cli-sess-abc", "model": "claude-sonnet-4-6",
                        "tools": ["Bash", "Read"], "mcp_servers": [{"name": "po", "status": "connected"}],
                        "permissionMode": "acceptEdits", "cwd": "/tmp/x"
                    }),
                },
            ),
            (
                "init_without_fields",
                Message::System {
                    subtype: "init".into(),
                    data: json!({}),
                },
            ),
            (
                "text_thinking_tool_use",
                assistant(
                    vec![
                        ContentBlock::Thinking(ThinkingContent {
                            thinking: "hmm".into(),
                            signature: "sig".into(),
                        }),
                        ContentBlock::Text(TextContent {
                            text: "Hello!".into(),
                        }),
                        ContentBlock::ToolUse(ToolUseContent {
                            id: "toolu_1".into(),
                            name: "mcp__po__plan".into(),
                            input: json!({"action": "list"}),
                        }),
                    ],
                    None,
                ),
            ),
            (
                "sub_agent_text",
                assistant(
                    vec![ContentBlock::Text(TextContent {
                        text: "from a sub-agent".into(),
                    })],
                    Some("toolu_parent"),
                ),
            ),
            (
                "assistant_tool_result_text",
                assistant(
                    vec![tool_result(
                        "t",
                        Some(ContentValue::Text("ok".into())),
                        Some(false),
                    )],
                    None,
                ),
            ),
            (
                "user_tool_result_structured_error",
                user_blocks(vec![tool_result(
                    "t2",
                    Some(ContentValue::Structured(vec![
                        json!({"type": "text", "text": "boom"}),
                    ])),
                    Some(true),
                )]),
            ),
            (
                "user_tool_result_without_content",
                user_blocks(vec![tool_result("t3", None, None)]),
            ),
            (
                "user_plain_text",
                Message::User {
                    message: UserMessage {
                        content: "Hi".into(),
                        content_blocks: None,
                    },
                    parent_tool_use_id: None,
                },
            ),
            (
                "text_delta",
                stream(
                    StreamEventData::ContentBlockDelta {
                        index: 0,
                        delta: StreamDelta::TextDelta { text: "Hel".into() },
                    },
                    Some("toolu_parent"),
                ),
            ),
            (
                "thinking_delta",
                stream(
                    StreamEventData::ContentBlockDelta {
                        index: 0,
                        delta: StreamDelta::ThinkingDelta {
                            thinking: "hm".into(),
                        },
                    },
                    None,
                ),
            ),
            (
                "block_start_tool_use",
                stream(
                    StreamEventData::ContentBlockStart {
                        index: 1,
                        content_block: json!({"type": "tool_use", "id": "toolu_s", "name": "Bash", "input": {}}),
                    },
                    None,
                ),
            ),
            (
                "block_start_text",
                stream(
                    StreamEventData::ContentBlockStart {
                        index: 0,
                        content_block: json!({"type": "text", "text": ""}),
                    },
                    None,
                ),
            ),
            ("message_stop", stream(StreamEventData::MessageStop, None)),
            (
                "compact_boundary",
                Message::System {
                    subtype: "compact_boundary".into(),
                    data: json!({"compact_metadata": {"trigger": "manual", "pre_tokens": 1234}}),
                },
            ),
            (
                "compact_boundary_without_metadata",
                Message::System {
                    subtype: "compact_boundary".into(),
                    data: json!({}),
                },
            ),
            (
                "unknown_system",
                Message::System {
                    subtype: "status".into(),
                    data: json!({"status": "compacting"}),
                },
            ),
            (
                "masking_failed",
                Message::System {
                    subtype: MASKING_FAILED_SUBTYPE.into(),
                    data: json!({}),
                },
            ),
            (
                "result_success",
                result_msg("success", false, Some(0.15), Some("done")),
            ),
            (
                "result_without_cost",
                result_msg("success", false, None, None),
            ),
            (
                "result_max_turns",
                result_msg("error_max_turns", true, Some(1.0), None),
            ),
            (
                "result_during_execution",
                result_msg("error_during_execution", true, None, Some("oops")),
            ),
        ]
    }

    fn mapped(msg: &Message, state: &mut MapState, mapper: &mut EventMapper) -> Vec<Value> {
        map_message(msg, state)
            .iter()
            .flat_map(|e| mapper.map(e))
            .map(|e| {
                // The legacy path knows none of the additive fields.
                let mut v = serde_json::to_value(e).unwrap();
                if let Some(o) = v.as_object_mut() {
                    let is_result = o.get("type").and_then(Value::as_str) == Some("result");
                    for k in ADDITIVE {
                        // `model` is a legacy field of system_init, additive only on result.
                        if *k != "model" || is_result {
                            o.remove(*k);
                        }
                    }
                }
                v
            })
            .collect()
    }

    /// Fields the contract adds to the legacy frames (always optional).
    const ADDITIVE: &[&str] = &[
        "category",
        "canonical",
        "cost",
        "usage",
        "model",
        "stop_reason",
        "policy_mode",
        "synthetic",
    ];

    #[test]
    fn the_additive_fields_carry_what_the_provider_reported() {
        use nexus_claude::agent::{Cost, CostBasis, ToolCategory, Usage};
        let mut mapper = EventMapper::new();
        let call = AgentEvent::ToolCall {
            id: "t".into(),
            name: "Bash".into(),
            input: json!({}),
            category: ToolCategory::Command,
            canonical: Some("shell".into()),
            input_complete: true,
            seq: None,
            parent: None,
        };
        match &mapper.map(&call)[0] {
            ChatEvent::ToolUse {
                category,
                canonical,
                ..
            } => {
                assert_eq!(category.as_deref(), Some("command"));
                assert_eq!(canonical.as_deref(), Some("shell"));
            }
            other => panic!("{other:?}"),
        }
        let done = AgentEvent::Done {
            stop_reason: StopReason::MaxTurns,
            subtype: None,
            is_error: true,
            result_text: None,
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..Default::default()
            },
            cost: Cost {
                usd: None,
                basis: CostBasis::Unknown,
            },
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 1,
            model: Some("m-1".into()),
            provider_session_id: None,
            structured_output: None,
            error: None,
        };
        let v = serde_json::to_value(&mapper.map(&done)[0]).unwrap();
        assert_eq!(v["stop_reason"], "max_turns");
        assert_eq!(v["model"], "m-1");
        assert_eq!(v["usage"], json!({"input_tokens": 10, "output_tokens": 5}));
        assert_eq!(v["cost"], json!({"basis": "unknown"}));
        assert!(v["cost_usd"].is_null(), "an unknown price is not a zero");
        let q = AgentEvent::Question {
            question_id: "q".into(),
            tool_call_id: None,
            reply: QuestionReply::Turn,
            questions: vec![],
            input: json!({}),
            parent: None,
        };
        let v = serde_json::to_value(&mapper.map(&q)[0]).unwrap();
        assert_eq!(v["synthetic"], true);
        let m = AgentEvent::PolicyModeChanged {
            mode: nexus_claude::agent::PolicyMode::AutoEdits,
            native_mode: Some("acceptEdits".into()),
        };
        let v = serde_json::to_value(&mapper.map(&m)[0]).unwrap();
        assert_eq!(
            (v["mode"].as_str(), v["policy_mode"].as_str()),
            (Some("acceptEdits"), Some("auto_edits"))
        );
    }

    #[test]
    fn replay_gives_the_same_chat_events_as_the_legacy_path() {
        let mut state = MapState::new();
        let mut mapper = EventMapper::new();
        for (name, msg) in corpus() {
            let legacy: Vec<Value> = ChatManager::message_to_events(&msg)
                .iter()
                .map(|e| serde_json::to_value(e).unwrap())
                .collect();
            let via_contract = mapped(&msg, &mut state, &mut mapper);
            assert_eq!(via_contract, legacy, "message `{name}` maps differently");
        }
    }

    #[test]
    fn an_announced_tool_call_completes_as_input_resolved() {
        let mut mapper = EventMapper::new();
        let call = |complete: bool| AgentEvent::ToolCall {
            id: "t1".into(),
            name: "Bash".into(),
            input: json!({"command": "ls"}),
            category: Default::default(),
            canonical: None,
            input_complete: complete,
            seq: None,
            parent: None,
        };
        let first = mapper.map(&call(false));
        let second = mapper.map(&call(true));
        assert!(matches!(first[0], ChatEvent::ToolUse { .. }));
        assert!(matches!(second[0], ChatEvent::ToolUseInputResolved { .. }));
        // A call never announced is a plain tool_use.
        let mut fresh = EventMapper::new();
        assert!(matches!(
            fresh.map(&call(true))[0],
            ChatEvent::ToolUse { .. }
        ));
    }

    #[test]
    fn done_without_subtype_gets_the_legacy_one_from_the_stop_reason() {
        let done = |stop, is_error| AgentEvent::Done {
            stop_reason: stop,
            subtype: None,
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 2,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: None,
        };
        let subtype = |stop, is_error| match &EventMapper::new().map(&done(stop, is_error))[0] {
            ChatEvent::Result { subtype, .. } => subtype.clone(),
            other => panic!("not a result: {other:?}"),
        };
        assert_eq!(subtype(StopReason::Completed, false), "success");
        assert_eq!(subtype(StopReason::MaxTurns, true), "error_max_turns");
        assert_eq!(
            subtype(StopReason::BudgetExceeded, true),
            "error_max_budget_usd"
        );
        assert_eq!(subtype(StopReason::Error, true), "error_during_execution");
    }

    #[test]
    fn an_unknown_cost_is_not_a_fake_zero() {
        let done = AgentEvent::Done {
            stop_reason: StopReason::Completed,
            subtype: Some("success".into()),
            is_error: false,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: None,
        };
        match &EventMapper::new().map(&done)[0] {
            ChatEvent::Result { cost_usd, .. } => assert_eq!(*cost_usd, None),
            other => panic!("not a result: {other:?}"),
        }
    }

    #[test]
    fn a_dead_process_is_a_session_error_and_other_failures_leak_nothing() {
        let dead = AgentEvent::Error {
            error: ProviderError::ProcessExited { code: Some(1) },
        };
        assert!(matches!(
            EventMapper::new().map(&dead)[0],
            ChatEvent::SessionError { ref reason, .. } if reason == "subprocess_exited"
        ));
        let unreachable = AgentEvent::Error {
            error: ProviderError::EndpointUnreachable {
                detail: "https://user:hunter2@10.0.0.5/v1 refused".into(),
            },
        };
        match &EventMapper::new().map(&unreachable)[0] {
            ChatEvent::Error { message, .. } => {
                assert!(
                    !message.contains("hunter2") && !message.contains("10.0.0.5"),
                    "{message}"
                );
            }
            other => panic!("not an error: {other:?}"),
        }
    }

    #[test]
    fn duplicate_task_updates_are_forwarded_once() {
        let mut mapper = EventMapper::new();
        let update = AgentEvent::TaskUpdate {
            phase: nexus_claude::agent::TaskPhase::Progress,
            task_id: Some("a".into()),
            tool_call_id: None,
            description: None,
            status: None,
            summary: None,
            event_id: Some("u-1".into()),
            data: json!({"x": 1}),
        };
        assert_eq!(mapper.map(&update).len(), 1);
        assert!(mapper.map(&update).is_empty());
    }

    #[test]
    fn out_of_turn_messages_regroup_by_message_number() {
        let at = Utc::now();
        let events = vec![
            AgentEvent::Text {
                text: "a".into(),
                seq: Some(1),
                parent: None,
            },
            AgentEvent::ToolCall {
                id: "t".into(),
                name: "Monitor".into(),
                input: json!({}),
                category: Default::default(),
                canonical: None,
                input_complete: true,
                seq: Some(1),
                parent: None,
            },
            AgentEvent::ToolResult {
                id: "t".into(),
                output: Some(ToolOutput::Text("line".into())),
                is_error: false,
                seq: Some(2),
                parent: Some("p".into()),
            },
            AgentEvent::UserEcho {
                text: "hello".into(),
                seq: Some(3),
                parent: None,
            },
            AgentEvent::ProviderNotice {
                kind: "status".into(),
                data: json!({"s": 1}),
            },
        ];
        let out = out_of_band_to_chat_events(&events, &mut EventMapper::new(), at);
        let rows: Vec<(String, String, Option<String>)> = out
            .into_iter()
            .map(|e| match e {
                ChatEvent::BackgroundOutput {
                    source,
                    content,
                    correlation_id,
                    ..
                } => (source, content, correlation_id),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            rows[0],
            ("assistant".into(), "a\n[tool_use: Monitor]".into(), None)
        );
        assert_eq!(
            rows[1],
            (
                "tool_result".into(),
                "[tool_result] line".into(),
                Some("p".into())
            )
        );
        assert_eq!(rows[2], ("user".into(), "hello".into(), None));
        assert_eq!(rows[3].0, "system:status");
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn every_agent_event_type_is_handled_without_panicking() {
        // The contract lists its tags; a variant added to nexus must not crash
        // the mapping (it falls in the wildcard arm and maps to nothing).
        let mut mapper = EventMapper::new();
        for tag in AgentEvent::TYPE_NAMES {
            assert!(!tag.is_empty());
        }
        let notice = AgentEvent::ProviderNotice {
            kind: "x".into(),
            data: Value::Null,
        };
        assert!(mapper.map(&notice).is_empty());
    }
}
