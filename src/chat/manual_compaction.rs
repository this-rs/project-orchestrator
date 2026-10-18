//! `/compact [instructions]` on the agent engine (ISO3, nexus contract v6 `compaction_manual`).
//!
//! The same command as Claude Code, typed in the same composer:
//!
//! - a provider that compacts on its own when it reads the command (the Claude Code CLI,
//!   whatever its engine) receives `/compact` as a turn, as before: nothing here applies;
//! - a provider that declares `compaction_manual` (the native harness) is asked
//!   `AgentSession::compact(instructions)` instead of a turn: the summary request carries
//!   the instructions, nexus announces `compaction started` / `completed` (`trigger:
//!   manual`) out of turn, and the outcome is said on the thread;
//! - any other provider: a typed refusal on the thread (`compaction_refused`), never a
//!   turn that would hand the model a command it cannot run.
//!
//! While a turn runs, `/compact` is refused (`turn_in_progress`, the nexus code): it is
//! never queued behind the turn nor allowed to interrupt it.

use nexus_claude::agent::{CompactOutcome, ProviderError};

use super::provider::endpoint_guard::redact_error_body;
use super::types::ChatEvent;

/// The command, as Claude Code spells it.
const COMMAND: &str = "/compact";

/// Longest cause kept from the provider's own words when a compaction fails.
const CAUSE_MAX: usize = 400;

/// `Some(instructions)` when `text` is the `/compact` command (`None` instructions when
/// nothing follows it); `None` for any other message (`/compacting`, a sentence that
/// mentions `/compact`...).
pub(crate) fn compact_command(text: &str) -> Option<Option<String>> {
    let rest = text.trim().strip_prefix(COMMAND)?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim();
    Some((!rest.is_empty()).then(|| rest.to_string()))
}

/// What the thread says of a manual compaction that ran.
pub(crate) fn outcome_event(outcome: &CompactOutcome) -> ChatEvent {
    let tokens = |t: Option<u64>| t.map_or_else(|| "?".to_string(), |t| format!("~{t}"));
    let content = if outcome.compacted {
        format!(
            "Conversation compacted on request: {} message(s) summarised, {} → {} tokens.",
            outcome.messages_summarised,
            tokens(outcome.tokens_before),
            tokens(outcome.tokens_after),
        )
    } else {
        format!(
            "Nothing to compact: the conversation is too short to summarise ({} tokens). \
             The history is unchanged.",
            tokens(outcome.tokens_before),
        )
    };
    ChatEvent::SystemHint { content }
}

/// What the thread says of a `/compact` that did not run: refused (`compaction_refused`:
/// the provider cannot, or a turn is running) or failed (`compaction_failed`, the history
/// intact). `reason` carries the nexus code, for a client to switch on.
pub(crate) fn refusal_event(error: &ProviderError) -> ChatEvent {
    let (message, code, reason) = match error {
        ProviderError::Unsupported { capability } => (
            format!(
                "Error: This session cannot compact on request: its provider does not support \
                 `{capability}`. Nothing was compacted."
            ),
            "compaction_refused",
            capability.clone(),
        ),
        ProviderError::TurnInProgress => (
            "Error: A turn is running: /compact cannot run until it ends. Send it again once \
             the turn is over. Nothing was compacted."
                .to_string(),
            "compaction_refused",
            error.kind().to_string(),
        ),
        other => {
            // Redacted in full first (a credential split by the cut would escape), then
            // bounded; a cut is said.
            let full = redact_error_body(&other.to_string(), usize::MAX);
            let mut cause = redact_error_body(&full, CAUSE_MAX);
            if cause.len() < full.len() {
                cause.push('…');
            }
            (
                format!(
                    "Error: The conversation could not be compacted: {cause}. The history is \
                     intact, nothing of the conversation is lost."
                ),
                "compaction_failed",
                other.kind().to_string(),
            )
        }
    };
    ChatEvent::Error {
        message,
        parent_tool_use_id: None,
        code: Some(code.to_string()),
        reason: Some(reason),
        index: None,
        request_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_is_read_with_or_without_instructions() {
        assert_eq!(compact_command("/compact"), Some(None));
        assert_eq!(compact_command("  /compact  \n"), Some(None));
        assert_eq!(
            compact_command("/compact keep the API decisions"),
            Some(Some("keep the API decisions".to_string()))
        );
        assert_eq!(
            compact_command("/compact\nline one\nline two "),
            Some(Some("line one\nline two".to_string()))
        );
    }

    #[test]
    fn another_message_is_not_the_command() {
        for text in [
            "",
            "compact",
            "/compacting now",
            "/compact-all",
            "please /compact",
            "/model x",
        ] {
            assert_eq!(compact_command(text), None, "{text:?}");
        }
    }

    #[test]
    fn the_outcome_is_said_in_numbers() {
        let done = outcome_event(&CompactOutcome {
            compacted: true,
            tokens_before: Some(9000),
            tokens_after: Some(1200),
            messages_summarised: 14,
        });
        let ChatEvent::SystemHint { content } = done else {
            panic!("{done:?}")
        };
        assert!(content.contains("14 message(s)") && content.contains("~9000 → ~1200"));
        let nothing = outcome_event(&CompactOutcome {
            compacted: false,
            tokens_before: Some(300),
            tokens_after: None,
            messages_summarised: 0,
        });
        let ChatEvent::SystemHint { content } = nothing else {
            panic!("{nothing:?}")
        };
        assert!(content.starts_with("Nothing to compact"), "{content}");
    }

    #[test]
    fn a_refusal_is_typed_and_readable() {
        let unsupported = refusal_event(&ProviderError::unsupported("compaction_manual"));
        assert!(
            matches!(&unsupported, ChatEvent::Error { code: Some(c), reason: Some(r), message, .. }
            if c == "compaction_refused" && r == "compaction_manual" && message.contains("compaction_manual"))
        );
        let busy = refusal_event(&ProviderError::TurnInProgress);
        assert!(
            matches!(&busy, ChatEvent::Error { code: Some(c), reason: Some(r), .. }
            if c == "compaction_refused" && r == "turn_in_progress")
        );
        let failed = refusal_event(&ProviderError::Protocol {
            detail: "the summary came back empty".into(),
        });
        assert!(
            matches!(&failed, ChatEvent::Error { code: Some(c), reason: Some(r), message, .. }
            if c == "compaction_failed" && r == "protocol" && message.contains("came back empty"))
        );
    }
}
