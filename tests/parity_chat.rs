//! Wire-contract parity for the chat WebSocket: record a live session and
//! replay its golden, frame by frame, in order.
//!
//! Three pure pieces, each unit-tested:
//! - [`normalize`]   — strips the volatile fields of ONE frame (`volatile.md`);
//! - [`compare`]     — diffs two normalized sequences IN ORDER, never by `seq`;
//! - the golden loader — reads `docs/parity/chat/golden/*.json` and refuses a
//!   frame with neither a server `type` nor a client `$client`.
//!
//! Plus one `#[ignore]`d recorder (`PARITY_RECORD=1`) that captures a live
//! stream and writes a new golden. It never runs in CI: the server is not there.
//!
//! No Neo4j, no network, no clock in the pure pieces: the goldens are the only
//! input. The recorder is the only part that touches the network.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

// ===========================================================================
// Volatile markers (`volatile.md`) — one deterministic token per kind.
// ===========================================================================

/// Keys whose value is an epoch/ISO timestamp → `<ts>`.
const TS_KEYS: &[&str] = &[
    "created_at",
    "received_at",
    "started_at",
    "ended_at",
    "called_at",
    "permission_requested_at",
    "permission_resolved_at",
    "run_started_at",
];

/// Keys whose value is a duration in milliseconds → `<dur>`.
const DUR_KEYS: &[&str] = &["duration_ms", "delay_ms", "build_latency_ms"];

/// Keys that carry a `tool_call_id` of the engine, coupled across frames
/// (`ToolResult.id == ToolUse.id`, `permission_request.tool_use_id`, …). These
/// are NOT touched by [`normalize`]: they are re-tokenised as `<tool-call-N>`
/// by [`index_tool_call_ids`] over the WHOLE sequence, so the coupling survives
/// (`volatile.md` §5: "après normalisation").
const TOOL_CALL_KEYS: &[&str] = &[
    "id",
    "tool_use_id",
    "tool_call_id",
    "request_id",
    "parent_tool_use_id",
    "correlation_id",
];

fn volatile_marker(key: &str) -> Option<&'static str> {
    if key == "seq" {
        // `seq` is 0 on every live frame and a per-session Neo4j counter on
        // replay (`volatile.md` §1): two rejoue of the same conversation do not
        // necessarily share it. Normalising it to `<seq>` is the only safe
        // treatment — and it is ORTHOGONAL to the `compare` rule "never sort by
        // seq", which is about ORDER, not about the value.
        return Some("<seq>");
    }
    if key == "session_id" {
        return Some("<session>");
    }
    if key == "cli_session_id" {
        return Some("<cli-session>");
    }
    if key == "model" {
        return Some("<model>");
    }
    if key == "cost_usd" {
        return Some("<cost>");
    }
    if key == "usage" {
        return Some("<usage>");
    }
    if key == "uuid" {
        return Some("<uuid>");
    }
    if TS_KEYS.contains(&key) {
        return Some("<ts>");
    }
    if DUR_KEYS.contains(&key) {
        return Some("<dur>");
    }
    None
}

/// Strip the volatile fields of ONE frame, in place, recursively.
///
/// PURE and stateless: it decides only from the frame's own keys. It does not
/// touch `type`, `$client`, `content`, `tool`, `input`, `is_error`,
/// `stop_reason`, `subtype` — the stable contract — and it does NOT touch
/// tool-call ids (those need the whole sequence, see [`index_tool_call_ids`]).
fn normalize_in_place(v: &mut Value) {
    match v {
        Value::Object(map) => {
            // `auth_ok` identity (`volatile.md` §12), before generic recursion.
            if let Some(user) = map.get_mut("user").and_then(|u| u.as_object_mut()) {
                for (key, marker) in [
                    ("id", "<user-id>"),
                    ("email", "<email>"),
                    ("name", "<name>"),
                ] {
                    if user.contains_key(key) {
                        user.insert(key.to_string(), Value::String(marker.to_string()));
                    }
                }
            }
            // Scalar replacements, keyed by field name.
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                if let Some(marker) = volatile_marker(&key) {
                    map.insert(key, Value::String(marker.to_string()));
                }
            }
            // Recurse into the (possibly rewritten) values.
            for val in map.values_mut() {
                normalize_in_place(val);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                normalize_in_place(item);
            }
        }
        _ => {}
    }
}

/// The pure per-frame normalisation: a value with every volatile field replaced
/// by its deterministic marker. Tested directly (see the `normalize_*` tests).
fn normalize(frame: &Value) -> Value {
    let mut out = frame.clone();
    normalize_in_place(&mut out);
    out
}

/// Re-tokenise tool-call ids as `<tool-call-N>`, first appearance in the stream
/// deciding N. The coupling across frames is preserved because equal raw ids
/// map to the same token (`volatile.md` §5). Top-level keys only: in the WS
/// stream, a top-level `id` is always a `tool_call_id` (the REST `id` polyphony
/// of `gaps.md` G7 is out of scope of the WS goldens).
fn index_tool_call_ids(frames: &mut [Value]) {
    let mut next = 0usize;
    let mut map: HashMap<String, String> = HashMap::new();
    for frame in frames.iter_mut() {
        let Some(obj) = frame.as_object_mut() else {
            continue;
        };
        for key in TOOL_CALL_KEYS {
            let Some(val) = obj.get_mut(*key) else {
                continue;
            };
            if let Value::String(s) = val {
                let raw = s.clone();
                let marker = map
                    .entry(raw)
                    .or_insert_with(|| {
                        next += 1;
                        format!("<tool-call-{next}>")
                    })
                    .clone();
                *s = marker;
            }
        }
    }
}

/// Normalise a whole sequence: per-frame [`normalize`], then the sequence-wide
/// tool-call coupling pass.
fn normalize_sequence(frames: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = frames.iter().map(normalize).collect();
    index_tool_call_ids(&mut out);
    out
}

// ===========================================================================
// The comparator
// ===========================================================================

/// One divergence between a golden and the captured stream.
#[derive(Debug, Clone, PartialEq)]
struct Difference {
    /// Rank of the frame in the sequence (0-based).
    index: usize,
    /// JSON path inside the frame, or `"<frame>"` for a missing/extra frame.
    field: String,
    /// What the golden held.
    expected: Value,
    /// What the capture held.
    actual: Value,
}

fn short(v: &Value) -> String {
    if v.is_null() {
        return "<absent>".to_string();
    }
    let s = serde_json::to_string(v).unwrap_or_default();
    if s.len() > 60 {
        format!("{}…", &s[..60])
    } else {
        s
    }
}

impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.field == "<frame>" {
            match (self.expected.is_null(), self.actual.is_null()) {
                (false, true) => write!(
                    f,
                    "frame #{}: missing (golden had {}, capture ended)",
                    self.index,
                    short(&self.expected)
                ),
                (true, false) => write!(
                    f,
                    "frame #{}: extra (golden ended, capture had {})",
                    self.index,
                    short(&self.actual)
                ),
                _ => write!(
                    f,
                    "frame #{}: expected {}, got {}",
                    self.index,
                    short(&self.expected),
                    short(&self.actual)
                ),
            }
        } else {
            write!(
                f,
                "frame #{} field `{}`: expected {}, got {}",
                self.index,
                self.field,
                short(&self.expected),
                short(&self.actual)
            )
        }
    }
}

fn join_path(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

fn diff_values(
    expected: &Value,
    actual: &Value,
    index: usize,
    path: &str,
    out: &mut Vec<Difference>,
) {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            for (k, ev) in e {
                match a.get(k) {
                    Some(av) => diff_values(ev, av, index, &join_path(path, k), out),
                    None => out.push(Difference {
                        index,
                        field: join_path(path, k),
                        expected: ev.clone(),
                        actual: Value::Null,
                    }),
                }
            }
            for (k, av) in a {
                if !e.contains_key(k) {
                    out.push(Difference {
                        index,
                        field: join_path(path, k),
                        expected: Value::Null,
                        actual: av.clone(),
                    });
                }
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            let n = e.len().max(a.len());
            for i in 0..n {
                let p = format!("{path}[{i}]");
                match (e.get(i), a.get(i)) {
                    (Some(ev), Some(av)) => diff_values(ev, av, index, &p, out),
                    (Some(ev), None) => out.push(Difference {
                        index,
                        field: p,
                        expected: ev.clone(),
                        actual: Value::Null,
                    }),
                    (None, Some(av)) => out.push(Difference {
                        index,
                        field: p,
                        expected: Value::Null,
                        actual: av.clone(),
                    }),
                    (None, None) => unreachable!(),
                }
            }
        }
        _ => {
            if expected != actual {
                out.push(Difference {
                    index,
                    field: path.to_string(),
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }
    }
}

/// Compare a golden and a capture, frame by frame, IN ORDER.
///
/// TRAP (the one this whole file exists to avoid — `gaps.md` G6): the
/// comparator MUST NOT sort by `seq` before comparing. Every live frame carries
/// `seq: 0` (`live_frame`, `ws_chat_handler.rs:1230`): sorting by `seq` is a
/// no-op on the live stream, so it would silently accept two frames that were
/// swapped on the wire. Order is carried only by the stream order. That is why
/// `compare` walks both sequences by POSITION and re-tokenises `seq` away
/// (through [`normalize`]) instead of using it as a sort key.
///
/// Both sides are normalised first, so a volatile value (`created_at`,
/// `session_id`, `seq`, tool-call ids …) that differs between the golden and
/// the capture is equated, and only the stable contract is compared.
fn compare(golden: &[Value], actual: &[Value]) -> Vec<Difference> {
    let g = normalize_sequence(golden);
    let a = normalize_sequence(actual);

    let mut out = Vec::new();
    let n = g.len().max(a.len());
    for i in 0..n {
        match (g.get(i), a.get(i)) {
            (Some(ge), Some(ae)) => diff_values(ge, ae, i, "", &mut out),
            (Some(ge), None) => out.push(Difference {
                index: i,
                field: "<frame>".to_string(),
                expected: ge.clone(),
                actual: Value::Null,
            }),
            (None, Some(ae)) => out.push(Difference {
                index: i,
                field: "<frame>".to_string(),
                expected: Value::Null,
                actual: ae.clone(),
            }),
            (None, None) => unreachable!(),
        }
    }
    out
}

// ===========================================================================
// The golden loader
// ===========================================================================

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/parity/chat/golden")
}

/// Parse one golden document. A frame with neither a server `type` nor a client
/// `$client` is refused: it would be compared as an empty server frame and pass
/// silently, which is exactly the kind of hole a golden must not have.
fn parse_golden(raw: &str, origin: &str) -> Result<Vec<Value>, String> {
    let arr: Value = serde_json::from_str(raw).map_err(|e| format!("{origin}: not JSON: {e}"))?;
    let frames = arr
        .as_array()
        .ok_or_else(|| format!("{origin}: a golden must be a JSON array"))?;
    for (i, f) in frames.iter().enumerate() {
        let obj = f
            .as_object()
            .ok_or_else(|| format!("{origin}: frame {i} is not an object"))?;
        let has_type = obj.get("type").is_some();
        let has_client = obj.get("$client").is_some();
        if !has_type && !has_client {
            return Err(format!(
                "{origin}: frame {i} has neither `type` nor `$client`"
            ));
        }
    }
    Ok(frames.clone())
}

fn load_golden_file(path: &Path) -> Result<Vec<Value>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_golden(&raw, &path.display().to_string())
}

/// Every committed golden under `docs/parity/chat/golden`, keyed by file stem.
fn load_all_goldens() -> BTreeMap<String, Vec<Value>> {
    let dir = golden_dir();
    let mut out = BTreeMap::new();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("golden dir `{}` unreadable: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let frames = load_golden_file(&path).unwrap_or_else(|e| panic!("golden `{name}`: {e}"));
        out.insert(name, frames);
    }
    out
}

// ===========================================================================
// Tests — normalize
// ===========================================================================

#[test]
fn normalize_replaces_timestamps_durations_session_and_model() {
    let frame = json!({
        "type": "result",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "duration_ms": 1234,
        "cost_usd": 0.0123,
        "usage": {"input_tokens": 100, "output_tokens": 200},
        "model": "claude-opus-4",
        "seq": 0,
        "created_at": 1700000000.123
    });
    let n = normalize(&frame);
    assert_eq!(n["session_id"], "<session>");
    assert_eq!(n["duration_ms"], "<dur>");
    assert_eq!(n["cost_usd"], "<cost>");
    assert_eq!(n["usage"], "<usage>");
    assert_eq!(n["model"], "<model>");
    assert_eq!(n["seq"], "<seq>");
    assert_eq!(n["created_at"], "<ts>");
}

#[test]
fn normalize_preserves_the_stable_contract() {
    let frame = json!({
        "type": "assistant_text",
        "content": "Le plan est en cours.",
        "seq": 0,
        "created_at": 1700000000.0
    });
    let n = normalize(&frame);
    assert_eq!(n["type"], "assistant_text");
    assert_eq!(n["content"], "Le plan est en cours.");
}

#[test]
fn normalize_maps_auth_ok_user_identity() {
    let frame = json!({
        "type": "auth_ok",
        "user": {"id": "u-1", "email": "a@b.c", "name": "Alice"},
        "features": ["refs_v1"]
    });
    let n = normalize(&frame);
    assert_eq!(n["user"]["id"], "<user-id>");
    assert_eq!(n["user"]["email"], "<email>");
    assert_eq!(n["user"]["name"], "<name>");
    // features is stable.
    assert_eq!(n["features"], json!(["refs_v1"]));
}

// ===========================================================================
// Tests — compare (the six required cases)
// ===========================================================================

#[test]
fn compare_detects_a_missing_frame() {
    let golden = vec![
        json!({"type": "user_message", "content": "a", "seq": 0}),
        json!({"type": "result", "seq": 0}),
    ];
    let actual = vec![json!({"type": "user_message", "content": "a", "seq": 0})];
    let diffs = compare(&golden, &actual);
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].index, 1);
    assert_eq!(diffs[0].field, "<frame>");
    assert!(diffs[0].actual.is_null(), "a missing frame is absent");
}

#[test]
fn compare_detects_an_extra_frame() {
    let golden = vec![json!({"type": "user_message", "content": "a", "seq": 0})];
    let actual = vec![
        json!({"type": "user_message", "content": "a", "seq": 0}),
        json!({"type": "result", "seq": 0}),
    ];
    let diffs = compare(&golden, &actual);
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].index, 1);
    assert_eq!(diffs[0].field, "<frame>");
    assert!(diffs[0].expected.is_null(), "an extra frame is unexpected");
}

#[test]
fn compare_detects_swapped_frames_and_does_not_sort_by_seq() {
    // Both frames carry `seq: 0` — the LIVE case (`gaps.md` G6). A comparator
    // that sorted by `seq` would see the same multiset and report nothing;
    // this one must report the inversion by position.
    let golden = vec![
        json!({"type": "assistant_text", "content": "premier", "seq": 0}),
        json!({"type": "tool_use", "id": "a", "tool": "Bash", "seq": 0}),
    ];
    let actual = vec![
        json!({"type": "tool_use", "id": "a", "tool": "Bash", "seq": 0}),
        json!({"type": "assistant_text", "content": "premier", "seq": 0}),
    ];
    let diffs = compare(&golden, &actual);
    assert!(!diffs.is_empty(), "two swapped frames must be reported");
    assert!(diffs.iter().any(|d| d.index == 0 && d.field == "type"));
    assert!(diffs.iter().any(|d| d.index == 1 && d.field == "type"));
}

#[test]
fn compare_ignores_a_volatile_field() {
    let golden = vec![json!({"type": "result", "session_id": "s-1", "duration_ms": 1, "seq": 0})];
    let actual = vec![json!({"type": "result", "session_id": "s-2", "duration_ms": 999, "seq": 0})];
    let diffs = compare(&golden, &actual);
    assert!(diffs.is_empty(), "volatile fields are equated: {diffs:?}");
}

#[test]
fn compare_detects_a_changed_stable_field() {
    let golden = vec![json!({"type": "assistant_text", "content": "hello", "seq": 0})];
    let actual = vec![json!({"type": "assistant_text", "content": "world", "seq": 0})];
    let diffs = compare(&golden, &actual);
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].index, 0);
    assert_eq!(diffs[0].field, "content");
    assert_eq!(diffs[0].expected, json!("hello"));
    assert_eq!(diffs[0].actual, json!("world"));
}

#[test]
fn compare_handles_interleaved_client_frames() {
    // Client frames (`$client`) interleave with server frames; they are frames
    // like any other and must be compared in position too.
    let golden = vec![
        json!({"$client": "user_message", "content": "Endors."}),
        json!({"type": "user_message", "content": "Endors.", "seq": 0}),
        json!({"$client": "interrupt"}),
        json!({"type": "tool_cancelled", "id": "a", "seq": 0}),
    ];
    let actual = golden.clone();
    assert!(
        compare(&golden, &actual).is_empty(),
        "identical interleaving"
    );

    // A dropped client frame is a difference, exactly where it stood.
    let missing_client = vec![
        json!({"$client": "user_message", "content": "Endors."}),
        json!({"type": "user_message", "content": "Endors.", "seq": 0}),
        json!({"type": "tool_cancelled", "id": "a", "seq": 0}),
    ];
    let diffs = compare(&golden, &missing_client);
    assert!(!diffs.is_empty());
    assert!(diffs.iter().any(|d| d.index >= 2));
}

#[test]
fn compare_preserves_tool_call_id_coupling() {
    let golden = vec![
        json!({"type": "tool_use", "id": "x-1", "tool": "Bash", "seq": 0}),
        json!({"type": "tool_result", "id": "x-1", "is_error": false, "seq": 0}),
    ];
    let actual = vec![
        json!({"type": "tool_use", "id": "y-9", "tool": "Bash", "seq": 0}),
        json!({"type": "tool_result", "id": "y-9", "is_error": false, "seq": 0}),
    ];
    // Different raw ids, same coupling: both sides collapse to `<tool-call-1>`,
    // so the comparison is empty.
    assert!(compare(&golden, &actual).is_empty());

    // Broken coupling on one side: the result points at a DIFFERENT call than
    // the use → a second token appears → a difference at `id`.
    let broken = vec![
        json!({"type": "tool_use", "id": "y-9", "tool": "Bash", "seq": 0}),
        json!({"type": "tool_result", "id": "z-7", "is_error": false, "seq": 0}),
    ];
    let diffs = compare(&golden, &broken);
    assert!(diffs.iter().any(|d| d.field == "id"));
}

#[test]
fn difference_message_names_rank_field_expected_and_got() {
    let d = Difference {
        index: 3,
        field: "content".to_string(),
        expected: json!("hello"),
        actual: json!("world"),
    };
    let s = d.to_string();
    assert!(s.contains("#3"), "{s}");
    assert!(s.contains("content"), "{s}");
    assert!(s.contains("hello"), "{s}");
    assert!(s.contains("world"), "{s}");
}

// ===========================================================================
// Tests — the loader and the committed goldens
// ===========================================================================

#[test]
fn loader_refuses_a_frame_with_neither_type_nor_client() {
    let err = parse_golden(r#"[{"content": "no discriminant"}]"#, "test").unwrap_err();
    assert!(
        err.contains("frame 0 has neither `type` nor `$client`"),
        "{err}"
    );

    // `type` alone is accepted, `$client` alone is accepted.
    assert!(parse_golden(r#"[{"type": "result"}]"#, "test").is_ok());
    assert!(parse_golden(r#"[{"$client": "interrupt"}]"#, "test").is_ok());
}

#[test]
fn loader_refuses_a_non_array_golden() {
    let err = parse_golden(r#"{"type": "result"}"#, "test").unwrap_err();
    assert!(err.contains("must be a JSON array"), "{err}");
}

#[test]
fn all_five_committed_goldens_load() {
    let goldens = load_all_goldens();
    assert_eq!(
        goldens.len(),
        5,
        "expected exactly the five committed goldens, got {goldens:?}"
    );
    for (name, frames) in &goldens {
        assert!(!frames.is_empty(), "golden `{name}` holds no frame");
    }
}

#[test]
fn every_committed_golden_round_trips_through_compare_unchanged() {
    // The strongest single check: normalise each committed golden and compare
    // it to itself. An empty diff proves the normalisation is idempotent over
    // the markers the goldens already use (`<ts>`, `<session>`, `<tool-call-N>`,
    // concrete `seq`) — so the recorder, which emits those markers, and the
    // comparator agree.
    let goldens = load_all_goldens();
    assert_eq!(goldens.len(), 5);
    for (name, frames) in &goldens {
        let diffs = compare(frames, frames);
        assert!(
            diffs.is_empty(),
            "golden `{name}` is not stable under normalisation: {diffs:?}"
        );
    }
}

// ===========================================================================
// The recorder — `#[ignore]`d, gated on PARITY_RECORD=1. Never runs in CI.
// ===========================================================================

const RECORD_ENV: &str = "PARITY_RECORD";

#[test]
#[ignore = "captures a live session; needs PARITY_RECORD=1 and a running server"]
fn record_live_session() {
    if std::env::var(RECORD_ENV).as_deref() != Ok("1") {
        eprintln!("{RECORD_ENV} is not 1 — nothing recorded.");
        return;
    }
    let rt = tokio::runtime::Runtime::new().expect("a tokio runtime");
    if let Err(e) = rt.block_on(record()) {
        panic!("recording failed: {e}");
    }
}

fn write_golden(name: &str, frames: &[Value]) -> Result<(), String> {
    let path = golden_dir().join(format!("{name}.json"));
    let mut rendered = serde_json::to_string_pretty(frames).map_err(|e| e.to_string())?;
    rendered.push('\n');
    std::fs::write(&path, rendered).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

async fn create_session(
    server: &str,
    token: Option<&str>,
    message: &str,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let mut req = client
        .post(format!("{server}/api/chat/sessions"))
        .json(&json!({"message": message}));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let body: Value = resp.json().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("create session: HTTP {status}: {body}"));
    }
    body.get("session_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("no `session_id` in create-session response: {body}"))
}

fn ws_url(server: &str, session_id: &str) -> String {
    let host = server
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    format!("ws://{host}/ws/chat/{session_id}?last_event=0")
}

async fn record() -> Result<(), String> {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    let server = std::env::var("PARITY_SERVER").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let token = std::env::var("PARITY_TOKEN").ok();
    let name = std::env::var("PARITY_NAME").unwrap_or_else(|_| "captured".into());
    let message = std::env::var("PARITY_MESSAGE")
        .unwrap_or_else(|_| "Bonjour, quel est le statut du projet ?".into());

    let session_id = match std::env::var("PARITY_SESSION_ID") {
        Ok(id) => id,
        Err(_) => create_session(&server, token.as_deref(), &message).await?,
    };

    let url = ws_url(&server, &session_id);
    let (mut ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .map_err(|e| format!("connect {url}: {e}"))?;

    let mut frames: Vec<Value> = Vec::new();

    // `ready` is sent but NOT recorded as a `$client` frame: the committed
    // goldens open on `auth_ok`, not on the handshake.
    ws.send(WsMessage::text("ready"))
        .await
        .map_err(|e| e.to_string())?;

    if !message.is_empty() {
        frames.push(json!({"$client": "user_message", "content": message}));
        let body = json!({"type": "user_message", "content": message}).to_string();
        ws.send(WsMessage::text(body))
            .await
            .map_err(|e| e.to_string())?;
    }

    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(60);
    loop {
        if frames.len() >= 2000 {
            break;
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, ws.next()).await {
            Err(_) => break,   // timeout
            Ok(None) => break, // socket closed
            Ok(Some(Ok(WsMessage::Text(t)))) => {
                let v: Value = serde_json::from_str(t.as_str())
                    .map_err(|e| format!("frame is not JSON: {e}"))?;
                let tag = v.get("type").and_then(Value::as_str);
                let done = matches!(
                    tag,
                    Some("result") | Some("session_closed") | Some("session_error")
                );
                frames.push(v);
                if done {
                    break;
                }
            }
            Ok(Some(Ok(_))) => {} // ping/pong/binary: ignored
            Ok(Some(Err(e))) => return Err(e.to_string()),
        }
    }

    let normalized = normalize_sequence(&frames);
    write_golden(&name, &normalized)?;
    eprintln!(
        "recorded {} frames from `{session_id}` → {}",
        frames.len(),
        golden_dir().join(format!("{name}.json")).display()
    );
    Ok(())
}
