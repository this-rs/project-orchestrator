//! Golden transcriptions of `ChatManager::message_to_events` (B7).
//!
//! Each RECORDING (`<name>.jsonl`, one `Message` per line) is replayed through
//! the legacy transcription; the events it produces are frozen in
//! `<name>.golden.json`. The same recordings are meant to be replayed after the
//! migration to the agent engine: identical output is the frame-parity proof.
//!
//! WHAT A RECORDING IS — AND WHAT IT IS NOT
//! A line is a `Message` in its SERDE shape, i.e. exactly the input of
//! `message_to_events`. It is NOT a raw CLI line. Turning CLI JSON into a
//! `Message` is `nexus_claude::message_parser::parse_message`, and that module
//! is private (`mod message_parser;` in the SDK's `lib.rs`), so a backend test
//! cannot reuse it. The two shapes differ where the CLI sends `user.content` as
//! an ARRAY of blocks (the tool results) while `UserMessage.content` is a
//! `String` and the blocks live in `content_blocks`.
//! `every_recording_line_is_exactly_a_message_as_the_sdk_writes_it` pins this
//! down: a line must re-serialize to itself, so no field can drift unnoticed.
//!
//! No CLI, no network, no clock: the fixtures are the only input.
//!
//! Regenerate after a DELIBERATE change — read the diff before committing it:
//! `UPDATE_CHAT_EVENT_GOLDEN=1 cargo test --test chat_event_golden`
use project_orchestrator::neo4j::traits::*;

use nexus_claude::Message;
use project_orchestrator::chat::manager::ChatManager;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Committed recordings and goldens, relative to the crate root.
const FIXTURE_DIR: &str = "tests/fixtures/harness";

/// Set to `1` to (re)generate the `.golden.json` files instead of comparing.
const UPDATE_ENV: &str = "UPDATE_CHAT_EVENT_GOLDEN";

/// The `subtype` of the system message the vault emits when a CLI message
/// holding a secret could not be masked. `manager::MASKING_FAILED_SUBTYPE` is
/// `pub(crate)` and out of reach of an integration test, so the recording uses
/// the literal: if the constant changes, the golden goes red. That is intended.
const MASKING_FAILED_SUBTYPE: &str = "po_masking_failed";

/// Every recorded situation, with what it stands for on the wire.
const SEQUENCES: &[(&str, &str)] = &[
    (
        "simple_turn",
        "A plain text turn: init, two streamed deltas, the finalized text, the result.",
    ),
    (
        "tool_turn",
        "One tool call and its result: thinking, text, Bash, the CLI's tool_result, then the answer.",
    ),
    (
        "permission_turn",
        "A turn the CLI gated on a permission: the call is announced empty, finalized, then refused.",
    ),
    (
        "compaction_boundary",
        "An automatic compaction boundary between two assistant messages.",
    ),
    (
        "background_task",
        "A background sub-agent: task_started, task_progress, task_notification.",
    ),
    (
        "overloaded_529",
        "An overloaded provider (529): the whole turn failed, nothing streamed.",
    ),
    (
        "unmapped_and_neutralized_frames",
        "Frames the legacy transcription deliberately drops, plus a tool result carrying a citation.",
    ),
];

/// The six situations the plan names, which must all be recorded.
const REQUIRED: &[&str] = &[
    "simple_turn",
    "tool_turn",
    "permission_turn",
    "compaction_boundary",
    "background_task",
    "overloaded_529",
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn recording_path(name: &str) -> PathBuf {
    fixture_dir().join(format!("{name}.jsonl"))
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir().join(format!("{name}.golden.json"))
}

fn updating() -> bool {
    std::env::var(UPDATE_ENV).as_deref() == Ok("1")
}

/// The JSON lines of a recording, blank lines skipped.
fn lines_of(name: &str) -> Vec<(usize, String)> {
    let path = recording_path(name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "recording `{}` is missing or unreadable ({e}).\n\
             A recorded situation with no recording is a hole in the guard.",
            path.display()
        )
    });
    raw.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| (i + 1, l.to_string()))
        .collect()
}

/// One line, as a `Message`. A line that does not parse is a broken
/// recording: it must fail loudly, never be skipped.
fn message_of(path: &std::path::Path, line_no: usize, line: &str) -> (Value, Message) {
    let value: Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("`{}` line {line_no} is not JSON: {e}", path.display()));
    let msg: Message = serde_json::from_value(value.clone()).unwrap_or_else(|e| {
        panic!(
            "`{}` line {line_no} is not a `Message`: {e}\nline: {line}",
            path.display()
        )
    });
    (value, msg)
}

/// Every message of a recording, in order.
fn recording(name: &str) -> Vec<Message> {
    let path = recording_path(name);
    let messages: Vec<Message> = lines_of(name)
        .into_iter()
        .map(|(n, line)| message_of(&path, n, &line).1)
        .collect();
    assert!(
        !messages.is_empty(),
        "recording `{}` holds no message",
        path.display()
    );
    messages
}

/// What `message_to_events` produces for the whole recording.
fn transcribe(name: &str) -> Value {
    let messages = recording(name);
    let mut events: Vec<Value> = Vec::new();
    for msg in &messages {
        for event in ChatManager::message_to_events(msg) {
            events.push(serde_json::to_value(&event).expect("a ChatEvent serialises"));
        }
    }
    json!({
        "sequence": name,
        "recording_lines": messages.len(),
        "events": events,
    })
}

fn render(doc: &Value) -> String {
    let mut out = serde_json::to_string_pretty(doc).expect("a JSON value renders");
    // `serde_json` writes the citation separator (U+2060 WORD JOINER) literally,
    // where it is INVISIBLE: a diff that removed the neutralization would look
    // identical to one that kept it. Escape it so the guard stays reviewable.
    // The comparison is on `Value`, so this changes the rendering only.
    out = out.replace('\u{2060}', "\\u2060");
    out.push('\n');
    out
}

/// Every event of a recording, as `(type tag, full JSON)` pairs.
fn tagged_events(name: &str) -> Vec<(String, Value)> {
    let messages = recording(name);
    let mut out = Vec::new();
    for msg in &messages {
        for event in ChatManager::message_to_events(msg) {
            let v = serde_json::to_value(&event).expect("a ChatEvent serialises");
            let tag = v["type"].as_str().unwrap_or("<no type>").to_string();
            out.push((tag, v));
        }
    }
    out
}

// ===========================================================================
// The guard
// ===========================================================================

/// THE test: every recording still transcribes to its golden file.
#[test]
fn the_golden_transcriptions_are_what_message_to_events_produces() {
    let update = updating();
    let mut compared = 0usize;

    for (name, _) in SEQUENCES {
        let produced = transcribe(name);
        let path = golden_path(name);

        if update {
            std::fs::write(&path, render(&produced)).expect("writes the golden file");
            continue;
        }

        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "golden `{}` is missing or unreadable ({e}).\n\
                 Generate it with `{UPDATE_ENV}=1 cargo test --test chat_event_golden`, \
                 then READ the diff before committing it.",
                path.display()
            )
        });
        let golden: Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("golden `{}` is not valid JSON: {e}", path.display()));

        assert_eq!(
            produced,
            golden,
            "the recording `{name}` no longer transcribes to `{}`.\n\
             If the change is deliberate, regenerate with \
             `{UPDATE_ENV}=1 cargo test --test chat_event_golden` and commit the diff \
             WITH the code change.",
            path.display()
        );
        compared += 1;
    }

    if !update {
        assert!(compared > 0, "no golden file was actually compared");
    }
}

/// A golden with no recording, or a recording with no golden, is a hole.
#[test]
fn the_recordings_and_the_golden_files_agree() {
    if updating() {
        // Another test of this same run is writing the goldens: the directory
        // cannot be in agreement yet, and judging it here would be a race.
        return;
    }

    let mut on_disk: Vec<String> = match std::fs::read_dir(fixture_dir()) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".jsonl") || n.ends_with(".golden.json"))
            .collect(),
        Err(e) => panic!("fixture directory `{FIXTURE_DIR}` is unreadable: {e}"),
    };
    on_disk.sort();

    let mut expected: Vec<String> = Vec::new();
    for (name, _) in SEQUENCES {
        expected.push(format!("{name}.jsonl"));
        expected.push(format!("{name}.golden.json"));
    }
    expected.sort();

    assert_eq!(
        on_disk, expected,
        "`{FIXTURE_DIR}` and the recorded situations disagree \
         (a stale file, a recording with no golden, or a golden with no recording)"
    );
}

/// The six situations the plan names must all be recorded.
#[test]
fn the_six_wire_situations_are_recorded() {
    let names: Vec<&str> = SEQUENCES.iter().map(|(n, _)| *n).collect();
    for expected in REQUIRED {
        assert!(
            names.contains(expected),
            "situation `{expected}` is not recorded; recorded: {names:?}"
        );
    }
}

/// A line must be EXACTLY what serde writes for the `Message` it parses into.
/// One extra field, one renamed field, one field serde would drop, and this
/// goes red: a recording cannot silently stop matching the SDK's shape.
#[test]
fn every_recording_line_is_exactly_a_message_as_the_sdk_writes_it() {
    for (name, _) in SEQUENCES {
        let path = recording_path(name);
        for (line_no, line) in lines_of(name) {
            let (value, message) = message_of(&path, line_no, &line);
            let rewritten = serde_json::to_value(&message).expect("a Message serialises");
            assert_eq!(
                rewritten,
                value,
                "`{}` line {line_no} is not what the SDK writes for that `Message` \
                 (a field is unknown, renamed, or optional in a way serde drops it)",
                path.display()
            );
        }
    }
}

/// A permission ask is NOT a `Message`: it travels on the control channel. The
/// legacy transcription must never invent one, and it publishes the call twice
/// under the same id (announced empty, then finalized).
#[test]
fn a_permission_ask_never_crosses_message_to_events() {
    let events = tagged_events("permission_turn");

    for forbidden in [
        "permission_request",
        "permission_decision",
        "ask_user_question",
    ] {
        assert!(
            !events.iter().any(|(tag, _)| tag == forbidden),
            "the legacy transcription emitted `{forbidden}`: a permission ask belongs to the \
             control channel, not to a `Message`. Tags seen: {:?}",
            events.iter().map(|(t, _)| t).collect::<Vec<_>>()
        );
    }

    let calls: Vec<&Value> = events
        .iter()
        .filter(|(tag, _)| tag == "tool_use")
        .map(|(_, v)| v)
        .collect();
    assert_eq!(
        calls.len(),
        2,
        "the announced and the finalized call are both published: {calls:?}"
    );
    assert_eq!(calls[0]["id"], calls[1]["id"], "same tool_use id");
    assert_eq!(
        calls[0]["input"],
        json!({}),
        "the announcement carries an empty input"
    );
    assert_eq!(
        calls[1]["input"],
        json!({"file_path": "notes.md", "content": "hi"}),
        "the finalized call carries the real input"
    );
}

/// ERR-1 gave the Result a typed `error_code`/`error_message` on the AGENT
/// path. The legacy transcription fills NEITHER. Frozen here so that wiring it
/// later is a visible change, not a silent one.
#[test]
fn an_overloaded_turn_keeps_no_typed_error_code_on_the_legacy_path() {
    let events = tagged_events("overloaded_529");
    let (_, result) = events
        .iter()
        .find(|(tag, _)| tag == "result")
        .expect("the failed turn still ends with a result");

    assert_eq!(result["subtype"], "error_during_execution");
    assert_eq!(result["is_error"], true);
    assert!(
        result["result_text"]
            .as_str()
            .is_some_and(|t| t.contains("529")),
        "the provider's own words are kept: {result}"
    );
    assert!(
        result.get("error_code").is_none(),
        "the legacy path carries no typed error code: {result}"
    );
    assert!(
        result.get("error_message").is_none(),
        "the legacy path carries no typed error message: {result}"
    );
}

/// The vault's masking-failure marker must surface as a visible error, never be
/// dropped. This also ties the recorded literal to [`MASKING_FAILED_SUBTYPE`]:
/// the constant is `pub(crate)`, so the recording cannot name it directly.
#[test]
fn the_masking_failure_marker_becomes_a_visible_error() {
    let recorded = lines_of("unmapped_and_neutralized_frames");
    assert!(
        recorded
            .iter()
            .any(|(_, line)| line.contains(MASKING_FAILED_SUBTYPE)),
        "the recording no longer carries the masking-failure marker `{MASKING_FAILED_SUBTYPE}`"
    );

    let events = tagged_events("unmapped_and_neutralized_frames");
    assert!(
        events.iter().any(|(tag, _)| tag == "error"),
        "the masking-failure marker must surface as an error: {:?}",
        events.iter().map(|(t, _)| t).collect::<Vec<_>>()
    );
}
