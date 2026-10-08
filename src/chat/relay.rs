//! Conversation relay (B-SW): what a session carries over when the user moves a
//! conversation to ANOTHER provider.
//!
//! Providers do not share a session format (a Claude Code session id means nothing
//! to the native harness, and a resume token is refused across families), so a
//! switch opens a NEW session on the target. What the target needs to continue the
//! conversation is rendered here from the stored events as plain text, and goes in
//! front of the user's next message.
//!
//! Nothing is summarised by a model (that would be a hidden, billed call): the text
//! is kept, tool calls are reduced to one line each with a bounded result, and when
//! the budget is exceeded the OLDEST turns are dropped and the omission is stated in
//! the relay itself. A relay never truncates in silence.

use super::types::ChatEvent;

/// Longest tool result kept, in characters.
const TOOL_RESULT_CHARS: usize = 400;
/// Longest tool input kept, in characters.
const TOOL_INPUT_CHARS: usize = 200;
/// Smallest budget honoured: below this a relay is not worth sending.
pub const MIN_BUDGET_CHARS: usize = 4_000;
/// Budget when the target's context window is unknown.
pub const DEFAULT_BUDGET_CHARS: usize = 48_000;
/// Largest budget, whatever the window.
pub const MAX_BUDGET_CHARS: usize = 400_000;

/// A rendered relay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relay {
    /// The text to put in front of the next message; empty when there was
    /// nothing worth carrying over.
    pub text: String,
    /// Conversation entries carried over.
    pub included: usize,
    /// Older entries left out to respect the budget (stated in `text`).
    pub omitted: usize,
}

/// The budget, in characters, for a target whose context window is `window_tokens`
/// (`None` = unknown). A relay may use at most 40 % of the window, at about four
/// characters per token, so the model keeps room to answer.
pub fn budget_for_window(window_tokens: Option<u64>) -> usize {
    match window_tokens {
        Some(tokens) => {
            let chars = tokens.saturating_mul(4).saturating_mul(4) / 10;
            usize::try_from(chars)
                .unwrap_or(MAX_BUDGET_CHARS)
                .clamp(MIN_BUDGET_CHARS, MAX_BUDGET_CHARS)
        }
        None => DEFAULT_BUDGET_CHARS,
    }
}

/// One conversation entry, already reduced to a line or a block.
struct Entry {
    text: String,
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}… [{} more characters]", text.chars().count() - max)
}

fn one_line(text: &str, max: usize) -> String {
    clip(&text.split_whitespace().collect::<Vec<_>>().join(" "), max)
}

fn result_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Reduces the stored events to the entries a relay carries: the user's messages,
/// the assistant's text, and one line per tool call with its bounded result.
/// Sub-agent (sidechain) text, thinking, permission traffic, streaming deltas and
/// every control event are left out.
fn entries(events: &[ChatEvent]) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    let mut calls: Vec<(String, usize)> = Vec::new();
    for event in events {
        match event {
            ChatEvent::UserMessage { content } if !content.trim().is_empty() => {
                out.push(Entry {
                    text: format!("[user]\n{}", content.trim()),
                });
            }
            ChatEvent::AssistantText {
                content,
                parent_tool_use_id: None,
            } if !content.trim().is_empty() => {
                out.push(Entry {
                    text: format!("[assistant]\n{}", content.trim()),
                });
            }
            ChatEvent::ToolUse {
                id,
                tool,
                input,
                parent_tool_use_id: None,
                ..
            } => {
                calls.push((id.clone(), out.len()));
                out.push(Entry {
                    text: format!(
                        "[tool {tool}] {}",
                        one_line(&input.to_string(), TOOL_INPUT_CHARS)
                    ),
                });
            }
            ChatEvent::ToolUseInputResolved { id, input, .. } => {
                if let Some((_, at)) = calls.iter().find(|(call, _)| call == id) {
                    if let Some(entry) = out.get_mut(*at) {
                        let tool = entry
                            .text
                            .strip_prefix("[tool ")
                            .and_then(|rest| rest.split(']').next())
                            .unwrap_or("tool")
                            .to_string();
                        entry.text = format!(
                            "[tool {tool}] {}",
                            one_line(&input.to_string(), TOOL_INPUT_CHARS)
                        );
                    }
                }
            }
            ChatEvent::ToolResult {
                id,
                result,
                is_error,
                parent_tool_use_id: None,
            } => {
                if let Some((_, at)) = calls.iter().find(|(call, _)| call == id) {
                    if let Some(entry) = out.get_mut(*at) {
                        let label = if *is_error { "error" } else { "result" };
                        entry.text.push_str(&format!(
                            "\n  → {label}: {}",
                            one_line(&result_text(result), TOOL_RESULT_CHARS)
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Renders the relay for `events`, from the provider the conversation was on to
/// the one it moves to, within `budget_chars`.
pub fn render_relay(events: &[ChatEvent], from: &str, to: &str, budget_chars: usize) -> Relay {
    let all = entries(events);
    if all.is_empty() {
        return Relay {
            text: String::new(),
            included: 0,
            omitted: 0,
        };
    }
    let budget = budget_chars.max(MIN_BUDGET_CHARS);
    // Newest first until the budget is spent; the newest entry is always kept
    // (clipped if it alone is larger than the budget).
    let mut kept: Vec<String> = Vec::new();
    let mut used = 0usize;
    for entry in all.iter().rev() {
        let len = entry.text.chars().count() + 2;
        if !kept.is_empty() && used + len > budget {
            break;
        }
        let text = if kept.is_empty() && len > budget {
            clip(&entry.text, budget.saturating_sub(2))
        } else {
            entry.text.clone()
        };
        used += text.chars().count() + 2;
        kept.push(text);
    }
    let included = kept.len();
    let omitted = all.len() - included;
    kept.reverse();

    let mut text = String::new();
    text.push_str(&format!(
        "<conversation_relay from=\"{from}\" to=\"{to}\">\n\
         This chat moved to you from another assistant engine. What follows is the earlier \
         conversation, as plain text: treat it as your own context, do not answer it again, \
         and continue from the user's next message below. Tool results are shortened; the \
         files and the project are exactly as the earlier turns left them.\n\n"
    ));
    if omitted > 0 {
        text.push_str(&format!(
            "[{omitted} earlier entr{} of this conversation {} left out to fit your context window]\n\n",
            if omitted == 1 { "y" } else { "ies" },
            if omitted == 1 { "was" } else { "were" },
        ));
    }
    text.push_str(&kept.join("\n\n"));
    text.push_str("\n</conversation_relay>");
    Relay {
        text,
        included,
        omitted,
    }
}

/// The first message of the new session: the relay, then the user's message.
pub fn compose_first_message(relay: &Relay, message: &str) -> String {
    if relay.text.is_empty() {
        return message.to_string();
    }
    format!("{}\n\n{}", relay.text, message)
}

/// What the model is sent: the relay (when the request carries one) in front of the
/// message. The conversation itself keeps showing `message` alone.
pub fn prefixed(relay: Option<&str>, message: &str) -> String {
    match relay {
        Some(text) if !text.is_empty() => format!("{text}\n\n{message}"),
        _ => message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(text: &str) -> ChatEvent {
        ChatEvent::UserMessage {
            content: text.into(),
        }
    }

    fn assistant(text: &str) -> ChatEvent {
        ChatEvent::AssistantText {
            content: text.into(),
            parent_tool_use_id: None,
        }
    }

    fn tool_use(id: &str, tool: &str, input: serde_json::Value) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: tool.into(),
            input,
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        }
    }

    fn tool_result(id: &str, result: serde_json::Value, is_error: bool) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result,
            is_error,
            parent_tool_use_id: None,
        }
    }

    #[test]
    fn a_conversation_is_carried_in_order_with_its_speakers() {
        let relay = render_relay(
            &[user("fix the bug"), assistant("looking"), user("thanks")],
            "claude-code",
            "local",
            DEFAULT_BUDGET_CHARS,
        );
        assert_eq!((relay.included, relay.omitted), (3, 0));
        let a = relay.text.find("[user]\nfix the bug").unwrap();
        let b = relay.text.find("[assistant]\nlooking").unwrap();
        let c = relay.text.find("[user]\nthanks").unwrap();
        assert!(a < b && b < c, "{}", relay.text);
        assert!(relay
            .text
            .starts_with("<conversation_relay from=\"claude-code\" to=\"local\">"));
        assert!(relay.text.ends_with("</conversation_relay>"));
    }

    #[test]
    fn a_tool_call_is_one_entry_with_its_bounded_result() {
        let long = "x".repeat(TOOL_RESULT_CHARS * 3);
        let relay = render_relay(
            &[
                tool_use("t1", "Bash", json!({"command": "ls"})),
                tool_result("t1", json!(long), false),
            ],
            "a",
            "b",
            DEFAULT_BUDGET_CHARS,
        );
        assert_eq!(relay.included, 1, "call and result are ONE entry");
        assert!(relay.text.contains("[tool Bash]"));
        assert!(relay.text.contains("→ result:"));
        assert!(
            relay.text.contains("more characters]"),
            "the clip is stated"
        );
        assert!(!relay.text.contains(&"x".repeat(TOOL_RESULT_CHARS + 50)));
    }

    #[test]
    fn a_failed_tool_call_says_error() {
        let relay = render_relay(
            &[
                tool_use("t1", "Read", json!({"file_path": "/nope"})),
                tool_result("t1", json!("no such file"), true),
            ],
            "a",
            "b",
            DEFAULT_BUDGET_CHARS,
        );
        assert!(
            relay.text.contains("→ error: no such file"),
            "{}",
            relay.text
        );
    }

    #[test]
    fn a_resolved_tool_input_replaces_the_empty_one() {
        let relay = render_relay(
            &[
                tool_use("t1", "Edit", json!({})),
                ChatEvent::ToolUseInputResolved {
                    id: "t1".into(),
                    input: json!({"file_path": "src/a.rs"}),
                    parent_tool_use_id: None,
                },
            ],
            "a",
            "b",
            DEFAULT_BUDGET_CHARS,
        );
        assert!(relay.text.contains("src/a.rs"), "{}", relay.text);
    }

    #[test]
    fn thinking_sidechains_and_control_events_are_left_out() {
        let relay = render_relay(
            &[
                user("hello"),
                ChatEvent::Thinking {
                    content: "secret reasoning".into(),
                    parent_tool_use_id: None,
                },
                ChatEvent::AssistantText {
                    content: "from a sub-agent".into(),
                    parent_tool_use_id: Some("parent".into()),
                },
                ChatEvent::SystemHint {
                    content: "a hint".into(),
                },
                assistant("hi"),
            ],
            "a",
            "b",
            DEFAULT_BUDGET_CHARS,
        );
        assert_eq!(relay.included, 2);
        for leaked in ["secret reasoning", "from a sub-agent", "a hint"] {
            assert!(
                !relay.text.contains(leaked),
                "{leaked} leaked: {}",
                relay.text
            );
        }
    }

    #[test]
    fn an_empty_conversation_gives_an_empty_relay_and_the_message_is_sent_as_is() {
        let relay = render_relay(&[], "a", "b", DEFAULT_BUDGET_CHARS);
        assert_eq!(relay.text, "");
        assert_eq!(compose_first_message(&relay, "hello"), "hello");
    }

    #[test]
    fn over_budget_the_oldest_entries_go_and_the_relay_says_so() {
        let filler = "w".repeat(1_000);
        let events: Vec<ChatEvent> = (0..40)
            .map(|i| user(&format!("turn-{i:02} {filler}")))
            .collect();
        let relay = render_relay(&events, "a", "b", MIN_BUDGET_CHARS);
        assert!(relay.omitted > 0 && relay.included > 0);
        assert_eq!(relay.included + relay.omitted, 40);
        assert!(
            relay.text.contains("left out to fit your context window"),
            "stated"
        );
        assert!(relay.text.contains("turn-39"), "the newest turn is kept");
        assert!(
            !relay.text.contains("turn-00"),
            "the oldest turn is dropped"
        );
        assert!(
            relay.text.chars().count() < MIN_BUDGET_CHARS + 1_500,
            "bounded"
        );
    }

    #[test]
    fn a_single_entry_larger_than_the_budget_is_clipped_not_dropped() {
        let relay = render_relay(
            &[user(&"z".repeat(MIN_BUDGET_CHARS * 3))],
            "a",
            "b",
            MIN_BUDGET_CHARS,
        );
        assert_eq!(relay.included, 1);
        assert!(relay.text.contains("more characters]"));
    }

    #[test]
    fn the_budget_follows_the_window_within_bounds() {
        assert_eq!(budget_for_window(None), DEFAULT_BUDGET_CHARS);
        assert_eq!(budget_for_window(Some(10)), MIN_BUDGET_CHARS);
        assert_eq!(budget_for_window(Some(32_000)), 51_200);
        assert_eq!(budget_for_window(Some(10_000_000)), MAX_BUDGET_CHARS);
    }

    #[test]
    fn the_message_follows_the_relay() {
        let relay = render_relay(&[user("a")], "x", "y", DEFAULT_BUDGET_CHARS);
        let first = compose_first_message(&relay, "next question");
        assert!(first.ends_with("next question"));
        assert!(
            first.find("</conversation_relay>").unwrap() < first.find("next question").unwrap()
        );
    }
}
