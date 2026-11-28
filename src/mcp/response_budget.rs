//! Response budget for PO's own MCP tools — the lever PO actually owns.
//!
//! # The measurement this module answers
//!
//! Measured on 2026-11-28 (`scripts/geo_usage_probe.py --tools`, 745 Claude Code
//! sessions, 86 451 tool results): tool output is 40.4 % of the corpus' cumulative
//! re-read volume, and 51 % of that comes from results under 2 000 tokens — the
//! mass is a QUEUE OF SMALL RESULTS re-read hundreds of times, not a few huge ones.
//!
//! Split by tool, the volume names who can act. `Bash` (51.8 %) and `Read` (29.6 %)
//! belong to the CLI: on the Claude Code harness PO is NOT between the tool and the
//! model and cannot rewrite that history — saying otherwise would be a lever that
//! does not exist. What PO does own is its own MCP responses, 13.5 % of the volume:
//! `note` 5.1 %, `task` 4.6 %, `plan` 2.3 %, `project` 1.2 %, `chat` 1.0 %,
//! `decision` 0.8 %, `step` 0.4 %.
//!
//! Those responses were written for one turn and then re-read by every request of
//! the session: a `plan get` of 30 KB is paid once per later request even though
//! nobody reads most of it again.
//!
//! # What this does
//!
//! One response, three representations, chosen by size and never by guesswork:
//!
//! * **small** (≤ [`MAX_INLINE_BYTES`]) — returned untouched. 95 % of tool calls
//!   live here; a digest would cost more than it saves.
//! * **large** — the WHOLE response is written, byte for byte, to the same
//!   content-addressed [`ArtifactStore`] `validate` uses, and the answer keeps its
//!   shape: every key, every short value (ids, statuses, names, counts) survives;
//!   only long strings ([`MAX_STRING_CHARS`]) and long arrays ([`MAX_ARRAY_ITEMS`])
//!   are elided, each replaced by a marker naming the address and how to read the
//!   rest.
//! * **`full: true`** — the caller asks for the raw payload inline; nothing is
//!   projected and nothing is stored.
//!
//! Nothing is lost and nothing is re-run: the address is the sha256 of the bytes,
//! so the handle IS the version (different content, different address), it stays
//! valid after a restart (the store is a directory), and
//! `validate(action: "read", artefact: <address>, from_line, to_line)` returns an
//! exact slice without executing anything again.
//!
//! # The address is not a capability
//!
//! A handle names bytes inside ONE [`ArtifactStore`]: the directory is the
//! authorization boundary. A caller that knows the sha256 but runs against another
//! store cannot resolve it — the read fails with a named `Missing`, never with bytes.
//! PO's launcher gives every session its own store (`PO_VALIDATE_STORE_DIR` under the
//! session's temporary directory, `nexus_tools::launch`), so a handle one session
//! received is not readable from another. `scripts/geo_mcp_bench.py` proves both
//! halves through the real MCP server, and `tests` below prove the local half.
//!
//! The policy is mirrored by `scripts/geo_usage_probe.py --budget`, which measures
//! it on real transcripts; `tests::policy_matches_the_probe` refuses a divergence.

use serde_json::{json, Value};

use crate::tool_digest::ArtifactStore;

/// Responses at or below this serialized size are returned untouched.
pub const MAX_INLINE_BYTES: usize = 8_000;

/// Longest string kept as is; longer ones keep their head and a marker.
pub const MAX_STRING_CHARS: usize = 400;

/// Longest array kept as is; the rest becomes one marker naming the omission.
pub const MAX_ARRAY_ITEMS: usize = 100;

/// What the projection did, for the log line and for the tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// `true` when the response was externalized.
    pub projected: bool,
    /// Serialized size before the projection.
    pub bytes_before: usize,
    /// Serialized size after it.
    pub bytes_after: usize,
    /// The address of the full payload, when one was written.
    pub artefact: Option<String>,
}

impl Outcome {
    /// Bytes this turn did not add to the context.
    pub fn bytes_saved(&self) -> usize {
        self.bytes_before.saturating_sub(self.bytes_after)
    }
}

/// The one-line instructions, written once at the top of the answer: repeating
/// them under every elided field would pay the address back on each one.
fn handle_hint(artefact: &str) -> String {
    format!(
        "read any range with validate(action: \"read\", artefact: \"{artefact}\", \
from_line: N, to_line: M) — byte-exact, nothing re-executed — or call the same \
tool with full: true"
    )
}

/// Project one tool response: untouched when small, externalized when large.
///
/// Deterministic: the same value and the same policy produce the same bytes, so
/// the same content always lands on the same address.
pub fn project(
    store: &ArtifactStore,
    tool: &str,
    value: &Value,
) -> Result<(Value, Outcome), String> {
    // The bytes that matter are the ones the MODEL sees: the rendered response, not a
    // compact JSON dump. Two reasons, both from the review of #1337: `validate read`
    // slices by LINES, and a `serde_json::to_vec` payload holds everything on ONE line
    // — a slice would then be the whole thing, and a slice over the budget would be
    // projected again. The rendered form is already one scalar per line.
    let rendered = crate::mcp::formatter::json_to_compact(value);
    let bytes = rendered.into_bytes();
    let before = bytes.len();
    let lines = crate::tool_digest::split_lines(&String::from_utf8_lossy(&bytes)).len();
    if before <= MAX_INLINE_BYTES {
        return Ok((
            value.clone(),
            Outcome {
                projected: false,
                bytes_before: before,
                bytes_after: before,
                artefact: None,
            },
        ));
    }
    let artefact = store.put(&bytes).map_err(|e| e.to_string())?;
    let address = artefact.as_str().to_string();
    let hint = handle_hint(&address);

    let mut projected = value.clone();
    elide(&mut projected, &address, lines);
    if let Some(map) = projected.as_object_mut() {
        map.insert(
            "_budget".to_string(),
            json!({
                "tool": tool,
                "bytes": before,
                "artefact": address,
                "lines": lines,
                "policy": format!(
                    "strings over {MAX_STRING_CHARS} chars and arrays over \
            {MAX_ARRAY_ITEMS} items are elided, keys and short values are intact; {hint}"
                ),
            }),
        );
    }
    let after = serde_json::to_vec(&projected)
        .map_err(|e| e.to_string())?
        .len();
    Ok((
        projected,
        Outcome {
            projected: true,
            bytes_before: before,
            bytes_after: after,
            artefact: Some(address),
        },
    ))
}

/// Replace what is only volume: long strings and long arrays.
fn elide(value: &mut Value, address: &str, lines: usize) {
    match value {
        Value::String(text) => {
            let chars = text.chars().count();
            if chars > MAX_STRING_CHARS {
                let head: String = text.chars().take(MAX_STRING_CHARS).collect();
                *text = format!(
                    "{head}… [{} chars elided — artefact {address}, {lines} lines]",
                    chars - MAX_STRING_CHARS
                );
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                elide(item, address, lines);
            }
            if items.len() > MAX_ARRAY_ITEMS {
                let omitted = items.len() - MAX_ARRAY_ITEMS;
                items.truncate(MAX_ARRAY_ITEMS);
                items.push(Value::String(format!(
                    "… [{omitted} more items elided — artefact {address}, {lines} lines]"
                )));
            }
        }
        Value::Object(map) => {
            for (_, child) in map.iter_mut() {
                elide(child, address, lines);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (ArtifactStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (ArtifactStore::new(dir.path()), dir)
    }

    fn big_plan() -> Value {
        json!({
            "plan": {"id": "b00d80c1-ec9b-4254-8947-70fd654a59e6", "status": "approved"},
            "tasks": (0..30).map(|i| json!({
                "id": format!("task-{i}"),
                "status": "pending",
                "description": format!("Tâche {i} : {}", "un texte de spécification très long ".repeat(40)),
            })).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn a_small_response_is_returned_untouched() {
        let (store, _dir) = store();
        let value = json!({"updated": true, "updated_fields": ["status"]});
        let (out, outcome) = project(&store, "task.update", &value).unwrap();
        assert_eq!(out, value);
        assert!(!outcome.projected);
        assert_eq!(outcome.artefact, None);
    }

    #[test]
    fn a_large_response_keeps_its_shape_and_hands_a_handle() {
        let (store, _dir) = store();
        let original = big_plan();
        let before = serde_json::to_vec(&original).unwrap().len();
        assert!(before > MAX_INLINE_BYTES, "{before}");

        let (out, outcome) = project(&store, "plan.get", &original).unwrap();
        assert!(outcome.projected);
        assert!(
            outcome.bytes_saved() > before / 2,
            "{} saved of {before}",
            outcome.bytes_saved()
        );

        // Every id survives: what the next call needs is never elided.
        let tasks = out["tasks"].as_array().unwrap();
        assert_eq!(tasks.len(), 30);
        assert_eq!(tasks[7]["id"], "task-7");
        assert_eq!(tasks[7]["status"], "pending");
        assert_eq!(out["plan"]["id"], "b00d80c1-ec9b-4254-8947-70fd654a59e6");
        // The long text is gone from the context, and says where it went.
        let desc = tasks[7]["description"].as_str().unwrap();
        assert!(
            desc.chars().count() < MAX_STRING_CHARS + 300,
            "{}",
            desc.len()
        );
        assert!(desc.contains("chars elided — artefact"), "{desc}");
        // The top-level address is named once, with the exact way to read it.
        let artefact = out["_budget"]["artefact"].as_str().unwrap();
        assert_eq!(Some(artefact.to_string()), outcome.artefact);
        assert!(out["_budget"]["policy"]
            .as_str()
            .unwrap()
            .contains("validate(action: \"read\""));
    }

    #[test]
    fn the_full_payload_is_retrievable_after_a_restart_without_re_running_anything() {
        let (store, dir) = store();
        let original = big_plan();
        // The artefact is the text the model WOULD have received, not a compact JSON
        // dump: one scalar per line, so a line range is a real range.
        let exact = crate::mcp::formatter::json_to_compact(&original).into_bytes();
        assert!(
            exact.iter().filter(|b| **b == b'\n').count() > 20,
            "the artefact is multi-line"
        );
        let (out, outcome) = project(&store, "plan.get", &original).unwrap();
        let address = outcome.artefact.unwrap();

        // A new process, the same directory: the handle still resolves.
        let reopened = ArtifactStore::new(dir.path());
        let reference = crate::tool_digest::Ref::from_hex(&address).unwrap();
        assert_eq!(reopened.len(&reference).unwrap(), exact.len());
        let whole = reopened.read_portion(&reference, 0, usize::MAX).unwrap();
        assert_eq!(
            whole.as_bytes(),
            exact.as_slice(),
            "byte-exact, not a summary"
        );
        assert!(
            !whole.contains("chars elided — artefact"),
            "the elision is not in the artefact"
        );
        // And a slice is a slice: a range of lines, no execution behind it.
        let first = reopened.read_portion(&reference, 0, 1).unwrap();
        assert_eq!(
            first.as_bytes(),
            exact.split(|b| *b == b'\n').next().unwrap()
        );
        assert!(out["_budget"]["artefact"].as_str().is_some());
    }

    #[test]
    fn another_caller_holding_the_same_address_is_refused() {
        let (mine, _dir_a) = store();
        let (theirs, _dir_b) = store();
        let original = big_plan();
        let (out, outcome) = project(&mine, "plan.get", &original).unwrap();
        let address = outcome.artefact.unwrap();
        assert!(out["_budget"]["artefact"].as_str().is_some());

        // Knowing the sha256 is not a capability: the store is the boundary.
        let reference = crate::tool_digest::Ref::from_hex(&address).unwrap();
        let refused = theirs.read_portion(&reference, 0, 5);
        assert!(refused.is_err(), "another caller must not read these bytes");
        assert!(mine.read_portion(&reference, 0, 5).is_ok(), "the owner can");
    }

    #[test]
    fn a_slice_is_bounded_and_the_whole_reassembles_exactly() {
        let (store, _dir) = store();
        let original = big_plan();
        let exact = crate::mcp::formatter::json_to_compact(&original).into_bytes();
        let (_, outcome) = project(&store, "plan.get", &original).unwrap();
        let reference =
            crate::tool_digest::Ref::from_hex(outcome.artefact.as_ref().unwrap()).unwrap();
        let lines = crate::tool_digest::split_lines(&String::from_utf8_lossy(&exact)).len();

        // One slice is small compared to the whole.
        let first = store.read_portion(&reference, 0, 10).unwrap();
        assert!(
            first.len() < exact.len() / 4,
            "a slice must be a slice: {}",
            first.len()
        );
        // Reading every slice in order rebuilds the artefact byte for byte.
        let mut rebuilt = String::new();
        let mut at = 0;
        while at < lines {
            let chunk = store
                .read_portion(&reference, at, (at + 50).min(lines))
                .unwrap();
            if at > 0 {
                rebuilt.push('\n');
            }
            rebuilt.push_str(&chunk);
            at += 50;
        }
        assert_eq!(
            rebuilt.as_bytes(),
            exact.as_slice(),
            "reconstruction is exact"
        );
    }

    #[test]
    fn long_arrays_are_capped_and_say_how_many_are_missing() {
        let (store, _dir) = store();
        // 1 500 small items: over `MAX_INLINE_BYTES`, every item short enough that
        // only the array cap can act.
        let value = json!({"items": (0..1_500).map(|i| json!({"id": i})).collect::<Vec<_>>()});
        let (out, outcome) = project(&store, "note.list", &value).unwrap();
        assert!(outcome.projected, "1 500 items is over the inline budget");
        let items = out["items"].as_array().unwrap();
        assert_eq!(
            items.len(),
            MAX_ARRAY_ITEMS + 1,
            "100 kept, one marker appended"
        );
        assert!(items
            .last()
            .unwrap()
            .as_str()
            .unwrap()
            .contains("more items elided"));
    }

    #[test]
    fn policy_matches_the_probe() {
        let script = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/geo_usage_probe.py"),
        )
        .expect("scripts/geo_usage_probe.py is in the repository");
        assert!(
            script.contains(&format!("MAX_INLINE_BYTES = {MAX_INLINE_BYTES}")),
            "MAX_INLINE_BYTES diverged"
        );
        assert!(
            script.contains(&format!("MAX_STRING_CHARS = {MAX_STRING_CHARS}")),
            "MAX_STRING_CHARS diverged"
        );
        assert!(
            script.contains(&format!("MAX_ARRAY_ITEMS = {MAX_ARRAY_ITEMS}")),
            "MAX_ARRAY_ITEMS diverged"
        );
    }
}
