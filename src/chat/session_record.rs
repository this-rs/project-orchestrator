//! The session record (`ChatSession` node): what both chat engines keep on it, by
//! the same rules.
//!
//! - `title` / `preview`: the user's first message as typed (never the
//!   `<po-refs>` / `<po-attachments>` blocks around it), cut at 80 / 200
//!   characters ([`title_and_preview`]). Set when the session is created, and by
//!   the backfills for a session that has none.
//! - `message_count`: 1 at creation (the opening message), +1 per later user
//!   message actually played ([`count_user_message`]). A system hint, a background
//!   output or a provider's `result` is not a message.
//! - `total_cost_usd`: what the session has cost so far ([`next_total_cost`]). The
//!   Claude Code CLI reports the session's total on each `result`; every other
//!   provider reports the cost of the turn, added up here. An unknown price leaves
//!   the total as it was: never an invented zero.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

/// Longest title, in characters (a longer one is cut at 77 and ends with `...`).
pub const TITLE_CHARS: usize = 80;
/// Longest preview, in characters (a longer one is cut at 197 and ends with `...`).
pub const PREVIEW_CHARS: usize = 200;

/// How long a write of the record may hold the turn of the agent engine: a store
/// that does not answer loses a figure, never the turn (as the post-stream steps
/// of the Claude Code engine, `post_stream::StepBudget`).
pub const RECORD_WRITE_BUDGET: Duration = Duration::from_secs(10);

fn cut(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        let kept: String = text.chars().take(max - 3).collect();
        format!("{}...", kept.trim_end())
    } else {
        text.to_string()
    }
}

/// The title and the preview of a session whose first user message is `stored`
/// (its stored form). `None` when the user typed nothing visible.
pub fn title_and_preview(stored: &str) -> Option<(String, String)> {
    let typed = crate::refs::turn::visible_text(stored);
    if typed.trim().is_empty() {
        return None;
    }
    Some((cut(&typed, TITLE_CHARS), cut(&typed, PREVIEW_CHARS)))
}

/// How a provider states the cost on the end of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostFigure {
    /// The session's total so far (the Claude Code CLI's `total_cost_usd`).
    SessionTotal,
    /// The cost of this turn alone (the agent contract's `done.cost`).
    Turn,
}

impl CostFigure {
    /// The figure a provider of `kind` (wire name of `ProviderKind`) reports.
    pub fn of_kind(kind: &str) -> Self {
        if kind == "claude_code" {
            Self::SessionTotal
        } else {
            Self::Turn
        }
    }
}

/// The session's total after a turn that reported `usd` (`None`: no price known).
pub fn next_total_cost(previous: Option<f64>, usd: Option<f64>, figure: CostFigure) -> Option<f64> {
    let usd = usd.filter(|v| v.is_finite())?;
    Some(match figure {
        CostFigure::SessionTotal => usd,
        CostFigure::Turn => previous.filter(|v| v.is_finite()).unwrap_or(0.0) + usd,
    })
}

/// Counts one more user message on the record of `session`.
pub async fn count_user_message(graph: &Arc<dyn GraphStore>, session: Uuid) -> anyhow::Result<()> {
    if let Some(node) = graph.get_chat_session(session).await? {
        graph
            .update_chat_session(
                session,
                None,
                None,
                Some(node.message_count + 1),
                None,
                None,
                None,
            )
            .await?;
    }
    Ok(())
}

/// Adds what a turn cost to the record of `session`. Nothing is written when the
/// turn's price is unknown.
pub async fn add_turn_cost(
    graph: &Arc<dyn GraphStore>,
    session: Uuid,
    usd: Option<f64>,
    figure: CostFigure,
) -> anyhow::Result<()> {
    if usd.is_none() {
        return Ok(());
    }
    if let Some(node) = graph.get_chat_session(session).await? {
        if let Some(total) = next_total_cost(node.total_cost_usd, usd, figure) {
            graph
                .update_chat_session(session, None, None, None, Some(total), None, None)
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_title_and_preview_are_the_typed_text_cut_at_80_and_200() {
        assert_eq!(
            title_and_preview("hello"),
            Some(("hello".to_string(), "hello".to_string()))
        );
        let long = "x".repeat(250);
        let (title, preview) = title_and_preview(&long).unwrap();
        assert_eq!(title.chars().count(), 80);
        assert!(title.ends_with("..."));
        assert_eq!(preview.chars().count(), 200);
        assert!(preview.ends_with("..."));
        assert_eq!(title_and_preview("   "), None);
        assert_eq!(title_and_preview(""), None);
    }

    #[test]
    fn the_title_never_carries_the_reference_block() {
        let task: crate::refs::types::EntityRef = serde_json::from_value(serde_json::json!({
            "kind": "task", "id": "00000000-0000-0000-0000-000000000001"
        }))
        .unwrap();
        let stored = crate::refs::block::encode("look at this", &[task]);
        assert_ne!(stored, "look at this", "the block is there");
        let (title, preview) = title_and_preview(&stored).unwrap();
        assert_eq!(title, "look at this");
        assert_eq!(preview, "look at this");
    }

    #[test]
    fn a_turn_cost_adds_up_a_session_total_replaces_and_an_unknown_price_changes_nothing() {
        use CostFigure::*;
        assert_eq!(next_total_cost(None, Some(0.5), Turn), Some(0.5));
        assert_eq!(next_total_cost(Some(0.5), Some(0.25), Turn), Some(0.75));
        assert_eq!(
            next_total_cost(Some(0.5), Some(0.75), SessionTotal),
            Some(0.75)
        );
        assert_eq!(next_total_cost(Some(0.5), None, Turn), None);
        assert_eq!(
            next_total_cost(None, Some(0.0), Turn),
            Some(0.0),
            "free: a real 0"
        );
        assert_eq!(next_total_cost(None, Some(f64::NAN), Turn), None);
        assert_eq!(CostFigure::of_kind("claude_code"), SessionTotal);
        assert_eq!(CostFigure::of_kind("native"), Turn);
    }
}
