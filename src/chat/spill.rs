//! Tool-result spilling.
//!
//! A huge tool result must not blow the model's context window. This module
//! turns an oversized result into two pieces:
//!
//! * the FULL text, persisted to the transcript store (the graph's
//!   `tool_result_spill` event) under a stable key, and
//! * a bounded, deterministic head/tail preview (with a marker naming the key
//!   and the byte count) that replaces the model-facing copy.
//!
//! The change is opt-in and budget-gated: a result whose text is at most
//! [`SpillConfig::budget_bytes`] is left untouched, a disabled budget never
//! spills, and a store that is unavailable (or a session without a transcript
//! id) keeps the result inline — the existing truncation path handles it, so
//! nothing is lost.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;

/// Event type of the transcript record that carries a spilled tool result.
pub(crate) const SPILL_EVENT_TYPE: &str = "tool_result_spill";

/// Environment variable that turns spilling off (any of `0`/`false`/`no`/`off`).
pub(crate) const SPILL_ENABLED_VAR: &str = "CHAT_TOOL_RESULT_SPILL";
/// Environment variable overriding the byte budget ([`SpillConfig::budget_bytes`]).
pub(crate) const SPILL_BUDGET_VAR: &str = "CHAT_TOOL_RESULT_SPILL_BUDGET_BYTES";

/// Default byte budget below which a tool result stays inline (16 KiB).
pub(crate) const DEFAULT_BUDGET_BYTES: usize = 16 * 1024;
/// Bytes kept from the head of a spilled result's preview.
pub(crate) const DEFAULT_HEAD_BYTES: usize = 1024;
/// Bytes kept from the tail of a spilled result's preview.
pub(crate) const DEFAULT_TAIL_BYTES: usize = 1024;
/// Cap on how many transcript events the accessor scans for one spill key.
#[cfg(test)]
const SPILL_SCAN_LIMIT: i64 = 1_000_000;

/// When a tool result's text is spilled.
#[derive(Debug, Clone, Copy)]
pub struct SpillConfig {
    /// Whether an oversized result is spilled at all (`false`: every result inline).
    pub enabled: bool,
    /// Byte budget: results whose text is at most this size stay inline.
    pub budget_bytes: usize,
    /// Bytes kept from the head of the preview.
    pub head_bytes: usize,
    /// Bytes kept from the tail of the preview.
    pub tail_bytes: usize,
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            budget_bytes: DEFAULT_BUDGET_BYTES,
            head_bytes: DEFAULT_HEAD_BYTES,
            tail_bytes: DEFAULT_TAIL_BYTES,
        }
    }
}

impl SpillConfig {
    /// The operator's configuration from the environment, with the default budget
    /// and spilling on when nothing (or nothing parseable) is set.
    pub(crate) fn from_env() -> Self {
        Self {
            enabled: env_bool(SPILL_ENABLED_VAR, true),
            budget_bytes: env_usize(SPILL_BUDGET_VAR, DEFAULT_BUDGET_BYTES),
            ..Self::default()
        }
    }
}

/// The text of a result that is safe to spill, or `None` when the result must
/// stay inline. Only a plain string, or a structured content whose blocks are
/// ALL text (a JSON string, or an object carrying a `text` string), is
/// spillable: anything else — an image block, a nested array, a JSON object —
/// has non-text content that replacing the whole result with a text preview
/// would irrecoverably drop. Those stay inline for the existing truncation path.
fn spillable_text(result: &serde_json::Value) -> Option<String> {
    match result {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => {
            let mut parts = Vec::with_capacity(blocks.len());
            for block in blocks {
                match block {
                    serde_json::Value::String(s) => parts.push(s.clone()),
                    serde_json::Value::Object(map) => {
                        let t = map.get("text").and_then(|t| t.as_str())?;
                        parts.push(t.to_string());
                    }
                    _ => return None,
                }
            }
            Some(parts.join("\n"))
        }
        _ => None,
    }
}

/// The bounded head/tail preview of a spilled result, with a marker naming the
/// key and the byte count: `head \n...(result spilled: <key>, N bytes)...\n tail`.
///
/// Deterministic for a given (text, config, key). When the head and tail already
/// cover the whole text (a budget smaller than the preview sizes), the text is
/// returned verbatim so a caller can tell the spill would not save anything.
pub(crate) fn build_preview(text: &str, cfg: &SpillConfig, key: &str) -> String {
    if cfg.head_bytes + cfg.tail_bytes >= text.len() {
        return text.to_string();
    }
    let head = truncate_head(text, cfg.head_bytes);
    let tail = truncate_tail(text, cfg.tail_bytes);
    format!(
        "{head}\n...(result spilled: {key}, {} bytes)...\n{tail}",
        text.len()
    )
}

/// Spills `result` when its text exceeds `cfg`'s budget. On a spill the full text
/// is persisted as a `tool_result_spill` transcript record under a fresh stable
/// key and `result` is replaced by the bounded preview. Small results, a disabled
/// budget, a preview no smaller than the text, or a store failure leave `result`
/// untouched (fail-soft). Only a plain string or a text-only shape is spilled: a
/// result with any non-text content (an image block, a nested array) stays inline
/// untouched, so nothing it carries is dropped — the existing truncation path
/// handles it.
pub(crate) async fn spill_if_oversized(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
    tool_result_id: &str,
    next_seq: &AtomicI64,
    result: &mut serde_json::Value,
    cfg: &SpillConfig,
) {
    if !cfg.enabled {
        return;
    }
    // A result with non-text content is never spilled: replacing it with a text
    // preview would drop the non-text blocks (images, arrays) for good.
    let Some(text) = spillable_text(result) else {
        return;
    };
    if text.len() <= cfg.budget_bytes {
        return;
    }
    let key = Uuid::new_v4().to_string();
    let preview = build_preview(&text, cfg, &key);
    // Never spill into a copy that is not strictly smaller than the text.
    if preview.len() >= text.len() {
        return;
    }
    let seq = next_seq.fetch_add(1, Ordering::SeqCst);
    match persist_spill(graph, session_uuid, tool_result_id, &key, &text, seq).await {
        Ok(()) => {
            tracing::debug!(
                tool_result_id,
                spill_key = %key,
                bytes = text.len(),
                "spilled an oversized tool result"
            );
            *result = serde_json::Value::String(preview);
        }
        Err(error) => {
            // Fail-soft: the result stays inline (the existing truncation path
            // handles it); nothing is lost.
            tracing::warn!(
                tool_result_id,
                error = %error,
                "could not spill an oversized tool result; keeping it inline"
            );
        }
    }
}

/// The payload of one spilled tool result, stored in the transcript under
/// [`SPILL_EVENT_TYPE`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpillRecord {
    key: String,
    tool_result_id: String,
    full_text: String,
}

/// Persists the full text of a spilled result under `key`.
async fn persist_spill(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Option<Uuid>,
    tool_result_id: &str,
    key: &str,
    full_text: &str,
    seq: i64,
) -> Result<()> {
    let Some(uuid) = session_uuid else {
        anyhow::bail!("cannot spill: the session has no transcript id");
    };
    let record = ChatEventRecord {
        id: Uuid::new_v4(),
        session_id: uuid,
        seq,
        event_type: SPILL_EVENT_TYPE.to_string(),
        data: serde_json::to_string(&SpillRecord {
            key: key.to_string(),
            tool_result_id: tool_result_id.to_string(),
            full_text: full_text.to_string(),
        })?,
        created_at: chrono::Utc::now(),
    };
    graph.store_chat_events(uuid, vec![record]).await
}

/// The full text spilled under `key`, if a record for it is in the transcript.
///
/// Test-only for now: nothing in production reads a spilled result back yet
/// (the model-facing copy is the preview). Wired to a caller when a read-back
/// path (e.g. a tool) exists; kept behind `cfg(test)` so the lib stays warning-free.
#[cfg(test)]
pub(crate) async fn read_spill(
    graph: &Arc<dyn GraphStore>,
    session_uuid: Uuid,
    key: &str,
) -> Result<Option<String>> {
    // `after_seq = -1` reads every event of the session, in sequence order.
    let events = graph
        .get_chat_events(session_uuid, -1, SPILL_SCAN_LIMIT)
        .await?;
    for event in events {
        if event.event_type != SPILL_EVENT_TYPE {
            continue;
        }
        let Ok(record) = serde_json::from_str::<SpillRecord>(&event.data) else {
            continue;
        };
        if record.key == key {
            return Ok(Some(record.full_text));
        }
    }
    Ok(None)
}

/// The head of `text`, at most `max` bytes, never splitting a UTF-8 char.
fn truncate_head(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The tail of `text`, at most `max` bytes, never splitting a UTF-8 char.
fn truncate_tail(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// A boolean from the environment, defaulting when unset or unparseable.
fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        }
        Err(_) => default,
    }
}

/// A positive number from the environment, defaulting when unset or unparseable.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use std::sync::atomic::AtomicI64;

    /// A config with a tiny budget, so a modest result spills in the tests.
    fn tiny() -> SpillConfig {
        SpillConfig {
            enabled: true,
            budget_bytes: 8,
            head_bytes: 4,
            tail_bytes: 4,
        }
    }

    #[test]
    fn a_small_result_stays_inline_in_the_decision() {
        let cfg = tiny();
        // 8 bytes is NOT over the budget (<= 8), so it stays inline.
        let preview = build_preview("12345678", &cfg, "k");
        assert_eq!(preview, "12345678", "head+tail cover the whole text");
    }

    #[test]
    fn the_preview_is_small_deterministic_and_marked() {
        let cfg = tiny();
        let text = format!("HEAD{}TAIL", "M".repeat(200)); // 208 bytes, over the 8-byte budget
        let a = build_preview(&text, &cfg, "k");
        let b = build_preview(&text, &cfg, "k");
        assert_eq!(a, b, "the preview is deterministic for the same key");
        assert!(
            a.len() < text.len(),
            "the preview is smaller: {} < {}",
            a.len(),
            text.len()
        );
        assert!(a.starts_with("HEAD"), "the head is kept: {a:?}");
        assert!(a.ends_with("TAIL"), "the tail is kept: {a:?}");
        assert!(
            a.contains(&format!("(result spilled: k, {} bytes)", text.len())),
            "the marker names the key and the byte count: {a:?}"
        );
    }

    #[test]
    fn the_preview_never_splits_a_multibyte_char() {
        let cfg = SpillConfig {
            enabled: true,
            budget_bytes: 8,
            head_bytes: 2, // cuts into the first multibyte char
            tail_bytes: 2,
        };
        let text = format!("a{}", "é".repeat(50)); // 'a' then 50 two-byte chars
        let preview = build_preview(&text, &cfg, "k");
        assert!(preview.len() < text.len(), "the preview is smaller");
        assert!(
            std::str::from_utf8(preview.as_bytes()).is_ok(),
            "the head/tail never split a UTF-8 char: {preview:?}"
        );
    }

    #[tokio::test]
    async fn a_small_result_is_inline_without_a_spill_record() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let uuid = Uuid::new_v4();
        let seq = AtomicI64::new(1);
        let mut result = serde_json::json!("short");
        spill_if_oversized(&graph, Some(uuid), "t1", &seq, &mut result, &tiny()).await;
        assert_eq!(result, serde_json::json!("short"), "small result untouched");
        let events = graph.get_chat_events(uuid, -1, 100).await.unwrap();
        assert!(
            events.iter().all(|e| e.event_type != SPILL_EVENT_TYPE),
            "no spill record for a small result"
        );
    }

    #[tokio::test]
    async fn an_oversized_result_spills_and_the_preview_is_bounded_and_marked() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let uuid = Uuid::new_v4();
        let seq = AtomicI64::new(1);
        let full = "A".repeat(1024);
        let mut result = serde_json::Value::String(full.clone());
        spill_if_oversized(&graph, Some(uuid), "t1", &seq, &mut result, &tiny()).await;

        let preview = result.as_str().expect("the result is a string preview");
        assert!(preview.len() < full.len(), "the preview is bounded");
        assert!(
            preview.contains("(result spilled:"),
            "the preview is marked: {preview}"
        );
        let events = graph.get_chat_events(uuid, -1, 100).await.unwrap();
        let spills: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == SPILL_EVENT_TYPE)
            .collect();
        assert_eq!(spills.len(), 1, "one spill record was persisted");
    }

    #[tokio::test]
    async fn the_full_text_round_trips_from_the_store() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let uuid = Uuid::new_v4();
        let seq = AtomicI64::new(1);
        let full = "the complete oversized output\n".repeat(64);
        let mut result = serde_json::Value::String(full.clone());
        spill_if_oversized(&graph, Some(uuid), "t1", &seq, &mut result, &tiny()).await;

        let preview = result.as_str().unwrap();
        let key = spill_key_from_preview(preview).expect("a spill key in the preview");
        let stored = read_spill(&graph, uuid, key).await.unwrap();
        assert_eq!(
            stored.as_deref(),
            Some(full.as_str()),
            "the full text is intact"
        );
    }

    #[tokio::test]
    async fn a_disabled_budget_keeps_the_result_inline() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let uuid = Uuid::new_v4();
        let seq = AtomicI64::new(1);
        let cfg = SpillConfig {
            enabled: false,
            ..tiny()
        };
        let mut result = serde_json::Value::String("A".repeat(1024));
        spill_if_oversized(&graph, Some(uuid), "t1", &seq, &mut result, &cfg).await;
        assert!(result.as_str().unwrap().len() == 1024, "kept inline");
        let events = graph.get_chat_events(uuid, -1, 100).await.unwrap();
        assert!(events.iter().all(|e| e.event_type != SPILL_EVENT_TYPE));
    }

    #[tokio::test]
    async fn a_session_without_a_transcript_id_falls_back_inline() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let seq = AtomicI64::new(1);
        let full = "A".repeat(1024);
        let mut result = serde_json::Value::String(full.clone());
        // `session_uuid: None` makes the persist fail; the result must stay inline.
        spill_if_oversized(&graph, None, "t1", &seq, &mut result, &tiny()).await;
        assert_eq!(result, serde_json::Value::String(full), "fail-soft: inline");
    }

    #[test]
    fn only_a_plain_string_or_a_text_only_shape_is_spillable() {
        assert_eq!(
            spillable_text(&serde_json::json!("x")),
            Some("x".to_string())
        );
        assert_eq!(
            spillable_text(&serde_json::json!([
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"}
            ])),
            Some("a\nb".to_string())
        );
        // Any non-text block (an image, a JSON object, null, a nested array)
        // makes the result non-spillable: spilling would drop it.
        assert_eq!(
            spillable_text(&serde_json::json!([
                {"text": "a"},
                {"type": "image", "source": {"data": "iVBOR"}}
            ])),
            None
        );
        assert_eq!(spillable_text(&serde_json::Value::Null), None);
        assert_eq!(spillable_text(&serde_json::json!({"n": 1})), None);
    }

    #[tokio::test]
    async fn an_oversized_result_with_an_image_block_is_not_spilled_and_keeps_its_shape() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let uuid = Uuid::new_v4();
        let seq = AtomicI64::new(1);
        // Oversized (the text block far exceeds the tiny budget) but structured:
        // an image block follows the text. Spilling would drop the image.
        let mut result = serde_json::json!([
            {"type": "text", "text": "A".repeat(1024)},
            {"type": "image", "source": {"data": "iVBORw0KGgo"}}
        ]);
        let before = result.clone();
        spill_if_oversized(&graph, Some(uuid), "t1", &seq, &mut result, &tiny()).await;
        assert_eq!(result, before, "the shape is kept untouched, byte for byte");
        let events = graph.get_chat_events(uuid, -1, 100).await.unwrap();
        assert!(
            events.iter().all(|e| e.event_type != SPILL_EVENT_TYPE),
            "no spill record for a result with non-text content"
        );
    }

    /// Parses the `(result spilled: <key>, N bytes)` marker out of a preview.
    fn spill_key_from_preview(preview: &str) -> Option<&str> {
        let start = preview.find("(result spilled: ")? + "(result spilled: ".len();
        let rest = &preview[start..];
        let end = rest.find(',')?;
        Some(&rest[..end])
    }
}
