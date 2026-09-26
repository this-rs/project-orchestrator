//! Session-lifetime consumer of the SDK control channel.
//!
//! # Why this exists
//!
//! The CLI answers a tool call only after its `PreToolUse` hook has been
//! answered by the host — us. Hook callbacks arrive as `hook_callback` control
//! requests on the SDK control channel. Until 2026-09-26 that channel was read
//! **only inside `stream_response`'s `select!` loop**, i.e. only while a turn
//! was streaming: the loop took the receiver out of the session at turn start
//! and parked it back in a mutex at turn end.
//!
//! Background `Task` subagents outlive the turn that spawned them. Once the
//! foreground turn ended, every tool call they made sent a `hook_callback`
//! that nobody read. The CLI timed out each one — "PreToolUse hook did not
//! respond before its timeout (host client may be unreachable)" — and **did not
//! execute the tool**. Five subagents spent twenty minutes producing nothing
//! but that message, and the next turn then dutifully answered requests that
//! had been dead for a quarter of an hour.
//!
//! # What it does
//!
//! One pump task per session owns the raw receiver for the whole session
//! lifetime. Hook callbacks are answered here, immediately and concurrently,
//! writing the response straight to the CLI's stdin. Everything else
//! (`can_use_tool` permission requests, control responses to our own
//! requests…) is forwarded unchanged to a second channel that
//! `stream_response` reads exactly as it read the raw one before — so the turn
//! loop is untouched, and its own hook branch simply never fires anymore.
//!
//! A slow hook (Neo4j, Meilisearch) no longer stalls the stream either: the
//! turn loop used to `await` each dispatch inline, serializing every hook
//! behind the previous one and behind the stream itself.
//!
//! # Limits, stated
//!
//! Non-hook messages are still only consumed while a turn streams. Between
//! turns they accumulate in the forward channel (same capacity as the raw one,
//! so no worse than before). A permission request raised by a background
//! subagent between turns therefore still waits for the next turn — that is a
//! separate problem (surfacing permissions with no active turn), not solved
//! here. Should the forward channel ever fill, the pump blocks on it and hook
//! answering stalls with it; at 8192 entries that takes an implausible idle
//! backlog, and it is logged.

use std::collections::HashMap;
use std::sync::Arc;

use nexus_claude::{
    build_hook_response_json, dispatch_hook_from_registry, is_hook_callback, HookCallback,
};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, info, warn};

/// The registry `InteractiveClient::hook_callbacks()` hands out: callback id →
/// callback. Shared with the client, which keeps registering into it.
pub type HookRegistry = Arc<RwLock<HashMap<String, Arc<dyn HookCallback>>>>;

/// Hooks answered concurrently for one session. Each hook can hit Neo4j and
/// Meilisearch; a workflow can fan out up to 16 subagents, and every one of
/// their tool calls raises a hook. Unbounded would hammer the graph on a
/// burst; fully serial is what timed out. Eight keeps a burst flowing without
/// turning one session into a load test.
const MAX_CONCURRENT_HOOKS: usize = 8;

/// Capacity of the forward channel. Mirrors `cli_channel_buffer_size(8192)`
/// in `ChatManager` so the turn loop sees exactly the backlog it saw before.
const FORWARD_CAPACITY: usize = 8192;

/// Start the pump for one session.
///
/// Takes the raw receiver obtained from
/// `InteractiveClient::take_sdk_control_receiver()` and returns the receiver
/// `stream_response` should read instead. `stdin_tx` is the lock-free stdin
/// sender (`clone_stdin_sender`), needed to answer hooks while a turn holds
/// the client lock; without it hooks are dispatched but cannot be answered,
/// which is logged loudly because it reproduces the original symptom.
///
/// The pump ends when the transport drops its sender (session closed) or when
/// the forward receiver is dropped.
pub fn spawn(
    session_id: String,
    mut raw_rx: mpsc::Receiver<serde_json::Value>,
    registry: HookRegistry,
    stdin_tx: Option<mpsc::Sender<String>>,
) -> mpsc::Receiver<serde_json::Value> {
    let (forward_tx, forward_rx) = mpsc::channel(FORWARD_CAPACITY);

    if stdin_tx.is_none() {
        warn!(
            session_id = %session_id,
            "Control pump started without a stdin sender: hook callbacks will be dispatched but never answered"
        );
    }

    tokio::spawn(async move {
        let gate = Arc::new(Semaphore::new(MAX_CONCURRENT_HOOKS));

        while let Some(msg) = raw_rx.recv().await {
            if !is_hook_callback(&msg) {
                if forward_tx.send(msg).await.is_err() {
                    debug!(session_id = %session_id, "Control pump: forward receiver dropped, stopping");
                    break;
                }
                continue;
            }

            let permit = match gate.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => break, // semaphore closed — never happens, but never spin
            };
            let registry = registry.clone();
            let stdin_tx = stdin_tx.clone();
            let session_id = session_id.clone();
            tokio::spawn(async move {
                let _permit = permit;
                answer_hook(&session_id, &msg, &registry, stdin_tx.as_ref()).await;
            });
        }

        debug!(session_id = %session_id, "Control pump ended: SDK control channel closed");
    });

    forward_rx
}

/// The CLI's request id lives either at the top level or under `request`,
/// depending on the control-protocol variant in use.
fn request_id_of(msg: &serde_json::Value) -> String {
    msg.get("request_id")
        .or_else(|| msg.get("request").and_then(|r| r.get("request_id")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Dispatch one hook callback and write its response to the CLI.
async fn answer_hook(
    session_id: &str,
    msg: &serde_json::Value,
    registry: &RwLock<HashMap<String, Arc<dyn HookCallback>>>,
    stdin_tx: Option<&mpsc::Sender<String>>,
) {
    let request_id = request_id_of(msg);

    let Some(result) = dispatch_hook_from_registry(msg, registry).await else {
        // Unknown callback id: the CLI will time this one out. It means the
        // registry and the CLI disagree on what was registered — worth a
        // warning, not worth inventing an answer.
        warn!(session_id = %session_id, request_id = %request_id, "Hook callback for an unknown callback_id — not answered");
        return;
    };

    let Some(tx) = stdin_tx else {
        warn!(session_id = %session_id, request_id = %request_id, "Hook callback dispatched but no stdin sender to answer it");
        return;
    };

    let response = build_hook_response_json(&request_id, &result);
    match tx.send(response).await {
        Ok(()) => {
            info!(session_id = %session_id, request_id = %request_id, "Hook callback answered")
        }
        Err(_) => {
            warn!(session_id = %session_id, request_id = %request_id, "Hook response could not be written: CLI stdin closed")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::{HookContext, HookInput, HookJSONOutput, SdkError, SyncHookJSONOutput};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::timeout;

    /// A hook that counts its calls and lets the tool through.
    struct Counting(Arc<AtomicUsize>);

    #[async_trait::async_trait]
    impl HookCallback for Counting {
        async fn execute(
            &self,
            _input: &HookInput,
            _tool_use_id: Option<&str>,
            _context: &HookContext,
        ) -> Result<HookJSONOutput, SdkError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(HookJSONOutput::Sync(SyncHookJSONOutput {
                continue_: Some(true),
                ..Default::default()
            }))
        }
    }

    fn registry_with(calls: &Arc<AtomicUsize>) -> HookRegistry {
        let mut map: HashMap<String, Arc<dyn HookCallback>> = HashMap::new();
        map.insert("cb-test".to_string(), Arc::new(Counting(calls.clone())));
        Arc::new(RwLock::new(map))
    }

    /// What the CLI sends when a registered hook fires (nested variant).
    fn hook_request(request_id: &str, callback_id: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {
                    "hook_event_name": "PreCompact",
                    "session_id": "sess",
                    "transcript_path": "/tmp/t.jsonl",
                    "cwd": "/",
                    "trigger": "auto"
                }
            }
        })
    }

    async fn next_response(stdin_rx: &mut mpsc::Receiver<String>) -> serde_json::Value {
        let raw = timeout(Duration::from_secs(2), stdin_rx.recv())
            .await
            .expect("a hook response within 2s")
            .expect("stdin channel open");
        serde_json::from_str(&raw).expect("hook response is JSON")
    }

    #[tokio::test]
    async fn hooks_are_answered_even_when_no_turn_reads_the_forward_channel() {
        // The incident: between turns nobody read the control channel, and
        // every hook a background subagent raised timed out. Here the forward
        // receiver is held but never read — hooks must still be answered.
        let (raw_tx, raw_rx) = mpsc::channel(16);
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(16);
        let calls = Arc::new(AtomicUsize::new(0));
        let _forward_rx = spawn("sess".into(), raw_rx, registry_with(&calls), Some(stdin_tx));

        for i in 0..5 {
            raw_tx
                .send(hook_request(&format!("req-{i}"), "cb-test"))
                .await
                .unwrap();
        }

        let mut answered = Vec::new();
        for _ in 0..5 {
            let response = next_response(&mut stdin_rx).await;
            assert_eq!(response["type"], "control_response");
            assert_eq!(response["response"]["subtype"], "success");
            answered.push(
                response["response"]["request_id"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
        }
        answered.sort();
        assert_eq!(answered, ["req-0", "req-1", "req-2", "req-3", "req-4"]);
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn non_hook_messages_are_forwarded_untouched_and_hooks_are_not() {
        let (raw_tx, raw_rx) = mpsc::channel(16);
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(16);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut forward_rx = spawn("sess".into(), raw_rx, registry_with(&calls), Some(stdin_tx));

        let permission = serde_json::json!({
            "type": "control_request",
            "request_id": "perm-1",
            "request": { "subtype": "can_use_tool", "tool_name": "Bash" }
        });
        raw_tx.send(hook_request("req-1", "cb-test")).await.unwrap();
        raw_tx.send(permission.clone()).await.unwrap();

        let forwarded = timeout(Duration::from_secs(2), forward_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(forwarded, permission);
        assert_eq!(
            next_response(&mut stdin_rx).await["response"]["request_id"],
            "req-1"
        );
        // The hook never reaches the turn loop.
        assert!(timeout(Duration::from_millis(200), forward_rx.recv())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn an_unknown_callback_does_not_block_the_next_hook() {
        let (raw_tx, raw_rx) = mpsc::channel(16);
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(16);
        let calls = Arc::new(AtomicUsize::new(0));
        let _forward_rx = spawn("sess".into(), raw_rx, registry_with(&calls), Some(stdin_tx));

        raw_tx
            .send(hook_request("req-unknown", "cb-missing"))
            .await
            .unwrap();
        raw_tx
            .send(hook_request("req-known", "cb-test"))
            .await
            .unwrap();

        assert_eq!(
            next_response(&mut stdin_rx).await["response"]["request_id"],
            "req-known"
        );
        assert!(
            timeout(Duration::from_millis(200), stdin_rx.recv())
                .await
                .is_err(),
            "an unknown callback must not be answered"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn the_pump_stops_when_the_transport_closes() {
        let (raw_tx, raw_rx) = mpsc::channel(16);
        let (stdin_tx, _stdin_rx) = mpsc::channel::<String>(16);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut forward_rx = spawn("sess".into(), raw_rx, registry_with(&calls), Some(stdin_tx));
        drop(raw_tx);
        assert!(timeout(Duration::from_secs(2), forward_rx.recv())
            .await
            .unwrap()
            .is_none());
    }
}
