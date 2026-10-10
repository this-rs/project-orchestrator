//! When each tool call really ran, as the engine saw it.
//!
//! ## Why
//!
//! A `tool_use` frame is stamped when the model ANNOUNCES the call (often before
//! its input has finished streaming) and a `tool_result` when the result comes
//! back. The span between the two also holds the time the user took to answer a
//! permission, and, for calls the engine runs one after another, the time the
//! previous calls ran. A trace drawn from those two times lies on exactly the
//! calls worth looking at.
//!
//! ## What is measured
//!
//! One [`ToolClock`] per session, fed from two doors that both engines have:
//!
//! - the **PreToolUse hook** of the shared hook table (`graph_hook_table`): the
//!   moment the engine takes the call up (the Claude Code CLI, or the agent
//!   engine through `GraphSessionHooks::before_tool`). On both engines it runs
//!   BEFORE the permission is asked;
//! - the **chat events** of the session: `tool_use` (announced),
//!   `permission_request` / `ask_user_question` (the engine waits for the user),
//!   `permission_decision` (the user answered), `tool_result` (the engine has the
//!   result).
//!
//! When the result arrives, the clock answers one [`ChatEvent::ToolTiming`] for
//! the call: every time it knows, in seconds since the epoch with the
//! milliseconds as the fraction (the unit of `created_at`), and `run_started_at`,
//! when the tool itself started running: the permission's answer when one was
//! asked, else the take-up by the engine, else the announcement. A permission
//! asked and never answered (a question answered by the result itself) leaves
//! `run_started_at` out rather than count the wait as run time.
//!
//! The legacy engine is fed by [`spawn_legacy_tap`] (a subscriber of the
//! session's broadcast, which persists and relays the timing); the agent engine
//! feeds its clock from its own `emit`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::broadcast;
use uuid::Uuid;

use super::types::ChatEvent;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::traits::GraphStore;

/// Seconds since the epoch, the milliseconds as the fraction (the chat wire's unit).
fn seconds(at: DateTime<Utc>) -> f64 {
    crate::api::ws_chat_handler::wire_seconds(at)
}

#[derive(Default)]
struct Call {
    tool: Option<String>,
    input: Option<Value>,
    parent: Option<String>,
    called_at: Option<DateTime<Utc>>,
    started_at: Option<DateTime<Utc>>,
    permission_requested_at: Option<DateTime<Utc>>,
    permission_resolved_at: Option<DateTime<Utc>>,
}

#[derive(Default)]
struct State {
    calls: HashMap<String, Call>,
    /// Permission request id → tool call id.
    requests: HashMap<String, String>,
}

/// The times of the tool calls of one session, until their result.
#[derive(Default)]
pub(crate) struct ToolClock {
    state: Mutex<State>,
}

fn registry() -> &'static Mutex<HashMap<String, Weak<ToolClock>>> {
    static CLOCKS: OnceLock<Mutex<HashMap<String, Weak<ToolClock>>>> = OnceLock::new();
    CLOCKS.get_or_init(Default::default)
}

impl ToolClock {
    /// The clock of a session: the hook and the event feeder of one session share
    /// it. It lives as long as one of them holds it.
    pub(crate) fn for_session(session_id: &str) -> Arc<ToolClock> {
        let mut clocks = registry().lock().unwrap_or_else(|e| e.into_inner());
        clocks.retain(|_, clock| clock.strong_count() > 0);
        if let Some(clock) = clocks.get(session_id).and_then(Weak::upgrade) {
            return clock;
        }
        let clock = Arc::new(ToolClock::default());
        clocks.insert(session_id.to_string(), Arc::downgrade(&clock));
        clock
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The engine takes the call up (PreToolUse). The first take-up counts.
    pub(crate) fn taken_up(&self, id: &str, tool: &str, input: &Value, at: DateTime<Utc>) {
        let mut state = self.state();
        let call = state.calls.entry(id.to_string()).or_default();
        call.started_at.get_or_insert(at);
        call.tool.get_or_insert_with(|| tool.to_string());
        if call.input.as_ref().is_none_or(is_empty) {
            call.input = Some(input.clone());
        }
    }

    /// Note what `event` says about a call; on its result, the call's timing.
    pub(crate) fn observe(&self, event: &ChatEvent, at: DateTime<Utc>) -> Option<ChatEvent> {
        let mut state = self.state();
        match event {
            ChatEvent::ToolUse {
                id,
                tool,
                input,
                parent_tool_use_id,
                ..
            } => {
                let call = state.calls.entry(id.clone()).or_default();
                call.called_at.get_or_insert(at);
                call.tool = Some(tool.clone());
                call.parent = parent_tool_use_id.clone();
                if !is_empty(input) {
                    call.input = Some(input.clone());
                }
                None
            }
            ChatEvent::ToolUseInputResolved { id, input, .. } => {
                if let Some(call) = state.calls.get_mut(id) {
                    call.input = Some(input.clone());
                }
                None
            }
            ChatEvent::PermissionRequest {
                id, tool, input, ..
            } => {
                if let Some(call_id) = asked_call(&state.calls, tool, input) {
                    if let Some(call) = state.calls.get_mut(&call_id) {
                        call.permission_requested_at = Some(at);
                    }
                    state.requests.insert(id.clone(), call_id);
                }
                None
            }
            ChatEvent::AskUserQuestion {
                id, tool_call_id, ..
            } => {
                let call = state.calls.entry(tool_call_id.clone()).or_default();
                call.permission_requested_at = Some(at);
                state.requests.insert(id.clone(), tool_call_id.clone());
                None
            }
            ChatEvent::PermissionDecision { id, .. } => {
                if let Some(call_id) = state.requests.remove(id) {
                    if let Some(call) = state.calls.get_mut(&call_id) {
                        call.permission_resolved_at = Some(at);
                    }
                }
                None
            }
            ChatEvent::ToolResult {
                id,
                parent_tool_use_id,
                ..
            }
            | ChatEvent::ToolCancelled {
                id,
                parent_tool_use_id,
            } => {
                let call = state.calls.remove(id)?;
                state.requests.retain(|_, call_id| call_id != id);
                Some(timing(id, call, parent_tool_use_id.clone(), at))
            }
            // The turn is over: a call without a result will not get one.
            ChatEvent::Result { .. } => {
                state.calls.clear();
                state.requests.clear();
                None
            }
            _ => None,
        }
    }
}

fn is_empty(input: &Value) -> bool {
    match input {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

/// The call a permission request is about: a call of that tool still waiting for
/// its result and not asked yet, the one with the same input first, else the one
/// the engine took up last.
fn asked_call(calls: &HashMap<String, Call>, tool: &str, input: &Value) -> Option<String> {
    let waiting: Vec<(&String, &Call)> = calls
        .iter()
        .filter(|(_, c)| c.permission_requested_at.is_none() && c.tool.as_deref() == Some(tool))
        .collect();
    waiting
        .iter()
        .find(|(_, c)| c.input.as_ref() == Some(input))
        .or_else(|| {
            waiting
                .iter()
                .max_by_key(|(_, c)| c.started_at.or(c.called_at))
        })
        .map(|(id, _)| (*id).clone())
}

fn timing(id: &str, call: Call, parent: Option<String>, ended_at: DateTime<Utc>) -> ChatEvent {
    let run_started_at = if call.permission_requested_at.is_some() {
        call.permission_resolved_at
    } else {
        call.started_at.or(call.called_at)
    };
    ChatEvent::ToolTiming {
        id: id.to_string(),
        called_at: call.called_at.map(seconds),
        started_at: call.started_at.map(seconds),
        permission_requested_at: call.permission_requested_at.map(seconds),
        permission_resolved_at: call.permission_resolved_at.map(seconds),
        run_started_at: run_started_at.map(seconds),
        ended_at: seconds(ended_at),
        parent_tool_use_id: parent.or(call.parent),
    }
}

/// A PreToolUse callback that tells the clock when the engine takes a call up,
/// then answers what the callback it wraps answers (or nothing, alone).
///
/// It WRAPS the table's PreToolUse callbacks rather than standing beside them: an
/// engine that calls only some of a table's callbacks (the scripted CLI of the
/// tests calls the first) still reaches the clock, and the first take-up counts.
pub(crate) struct ToolClockHook {
    clock: Arc<ToolClock>,
    inner: Option<Arc<dyn nexus_claude::HookCallback>>,
}

/// Puts the session's clock on every PreToolUse callback of `table`, or adds it
/// alone when the table has none (a runner session).
pub(crate) fn clock_the_table(table: &mut super::agent_hooks::HookTable, clock: Arc<ToolClock>) {
    let matchers = table.entry("PreToolUse".to_string()).or_default();
    for matcher in matchers.iter_mut() {
        for hook in matcher.hooks.iter_mut() {
            *hook = Arc::new(ToolClockHook {
                clock: Arc::clone(&clock),
                inner: Some(Arc::clone(hook)),
            });
        }
    }
    if matchers
        .iter()
        .all(|m| m.matcher.is_some() || m.hooks.is_empty())
    {
        matchers.push(nexus_claude::HookMatcher {
            matcher: None,
            hooks: vec![Arc::new(ToolClockHook { clock, inner: None })],
        });
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for ToolClockHook {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        tool_use_id: Option<&str>,
        context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        if let (nexus_claude::HookInput::PreToolUse(pre), Some(id)) = (input, tool_use_id) {
            self.clock
                .taken_up(id, &pre.tool_name, &pre.tool_input, Utc::now());
        }
        match &self.inner {
            Some(inner) => inner.execute(input, tool_use_id, context).await,
            None => Ok(nexus_claude::HookJSONOutput::Sync(
                nexus_claude::SyncHookJSONOutput {
                    continue_: Some(true),
                    ..Default::default()
                },
            )),
        }
    }
}

/// Feeds the clock of a legacy (Claude Code CLI) session from its broadcast, and
/// persists and relays each timing like the session's other events. It holds the
/// sender weakly: it ends with the session's channel.
pub(crate) fn spawn_legacy_tap(
    session_id: String,
    events_tx: &broadcast::Sender<ChatEvent>,
    nats: Option<Arc<crate::events::NatsEmitter>>,
    graph: Arc<dyn GraphStore>,
    next_seq: Arc<AtomicI64>,
) {
    let clock = ToolClock::for_session(&session_id);
    let mut rx = events_tx.subscribe();
    let tx = events_tx.downgrade();
    let uuid = Uuid::parse_str(&session_id).ok();
    tokio::spawn(async move {
        loop {
            let event = match rx.recv().await {
                Ok(event) => event,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            };
            let Some(timing) = clock.observe(&event, Utc::now()) else {
                continue;
            };
            if let Some(uuid) = uuid {
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: timing.event_type().to_string(),
                    data: serde_json::to_string(&timing).unwrap_or_default(),
                    created_at: Utc::now(),
                };
                let _ = graph.store_chat_events(uuid, vec![record]).await;
            }
            if let Some(nats) = &nats {
                nats.publish_chat_event(&session_id, timing.clone());
            }
            let Some(tx) = tx.upgrade() else { break };
            let _ = tx.send(timing);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn at(ms: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(ms).unwrap()
    }

    fn tool_use(id: &str, tool: &str, input: Value) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: tool.into(),
            input,
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        }
    }

    fn result(id: &str) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result: json!("ok"),
            is_error: false,
            parent_tool_use_id: None,
        }
    }

    fn timing_fields(event: ChatEvent) -> Value {
        assert_eq!(event.event_type(), "tool_timing");
        serde_json::to_value(event).unwrap()
    }

    #[test]
    fn a_call_without_permission_runs_from_its_take_up_to_its_result() {
        let clock = ToolClock::default();
        assert!(clock
            .observe(&tool_use("t1", "Read", json!({})), at(1_000))
            .is_none());
        clock.taken_up("t1", "Read", &json!({"file_path": "a"}), at(1_250));
        let t = timing_fields(clock.observe(&result("t1"), at(1_900)).unwrap());
        assert_eq!(t["id"], "t1");
        assert_eq!(t["called_at"], 1.0);
        assert_eq!(t["started_at"], 1.25);
        assert_eq!(t["run_started_at"], 1.25);
        assert_eq!(t["ended_at"], 1.9);
        assert!(t.get("permission_requested_at").is_none(), "{t}");
    }

    #[test]
    fn the_wait_for_a_permission_is_not_run_time() {
        let clock = ToolClock::default();
        let input = json!({"command": "ls"});
        clock.observe(&tool_use("t1", "Bash", input.clone()), at(1_000));
        clock.taken_up("t1", "Bash", &input, at(1_100));
        clock.observe(
            &ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Bash".into(),
                input: input.clone(),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
            at(1_200),
        );
        clock.observe(
            &ChatEvent::PermissionDecision {
                id: "req-1".into(),
                allow: true,
            },
            at(9_200),
        );
        let t = timing_fields(clock.observe(&result("t1"), at(9_500)).unwrap());
        assert_eq!(t["started_at"], 1.1);
        assert_eq!(t["permission_requested_at"], 1.2);
        assert_eq!(t["permission_resolved_at"], 9.2);
        assert_eq!(t["run_started_at"], 9.2);
        assert_eq!(t["ended_at"], 9.5);
    }

    #[test]
    fn a_permission_goes_to_the_call_with_the_same_input() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("a", "Bash", json!({"command": "ls"})), at(1_000));
        clock.observe(
            &tool_use("b", "Bash", json!({"command": "rm x"})),
            at(1_001),
        );
        clock.taken_up("b", "Bash", &json!({"command": "rm x"}), at(1_002));
        clock.observe(
            &ChatEvent::PermissionRequest {
                id: "r".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
            at(1_500),
        );
        let a = timing_fields(clock.observe(&result("a"), at(2_000)).unwrap());
        assert_eq!(a["permission_requested_at"], 1.5);
        let b = timing_fields(clock.observe(&result("b"), at(2_100)).unwrap());
        assert!(b.get("permission_requested_at").is_none(), "{b}");
    }

    #[test]
    fn a_question_answered_by_the_result_has_no_run_start() {
        let clock = ToolClock::default();
        clock.observe(&tool_use("q", "AskUserQuestion", json!({})), at(1_000));
        clock.observe(
            &ChatEvent::AskUserQuestion {
                id: "req".into(),
                tool_call_id: "q".into(),
                questions: json!([]),
                input: json!({}),
                parent_tool_use_id: None,
                synthetic: None,
            },
            at(1_100),
        );
        let t = timing_fields(clock.observe(&result("q"), at(5_000)).unwrap());
        assert_eq!(t["permission_requested_at"], 1.1);
        assert!(t.get("run_started_at").is_none(), "{t}");
    }

    #[test]
    fn a_result_of_an_unknown_call_and_the_end_of_a_turn_give_no_timing() {
        let clock = ToolClock::default();
        assert!(clock.observe(&result("never-seen"), at(1)).is_none());
        clock.observe(&tool_use("t", "Read", json!({})), at(2));
        clock.observe(
            &serde_json::from_value::<ChatEvent>(json!({
                "type": "result", "session_id": "s", "duration_ms": 0,
                "subtype": "success", "is_error": false
            }))
            .unwrap(),
            at(3),
        );
        assert!(clock.observe(&result("t"), at(4)).is_none());
    }

    struct Says(&'static str);

    #[async_trait::async_trait]
    impl nexus_claude::HookCallback for Says {
        async fn execute(
            &self,
            _input: &nexus_claude::HookInput,
            _tool_use_id: Option<&str>,
            _context: &nexus_claude::HookContext,
        ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
            Ok(nexus_claude::HookJSONOutput::Sync(
                nexus_claude::SyncHookJSONOutput {
                    reason: Some(self.0.to_string()),
                    ..Default::default()
                },
            ))
        }
    }

    fn pre_tool_use(tool: &str) -> nexus_claude::HookInput {
        nexus_claude::HookInput::PreToolUse(nexus_claude::PreToolUseHookInput {
            session_id: "s".into(),
            transcript_path: String::new(),
            cwd: "/w".into(),
            permission_mode: None,
            tool_name: tool.into(),
            tool_input: json!({"command": "ls"}),
            agent_id: None,
            agent_type: None,
        })
    }

    #[tokio::test]
    async fn the_first_pre_tool_use_callback_clocks_the_call_and_still_answers() {
        let clock = Arc::new(ToolClock::default());
        let mut table = super::super::agent_hooks::HookTable::new();
        table.insert(
            "PreToolUse".into(),
            vec![nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![Arc::new(Says("skill context"))],
            }],
        );
        clock_the_table(&mut table, Arc::clone(&clock));
        assert_eq!(table["PreToolUse"].len(), 1, "wrapped, not added");
        let first = &table["PreToolUse"][0].hooks[0];
        let ctx = nexus_claude::HookContext { signal: None };
        let out = first
            .execute(&pre_tool_use("Bash"), Some("t1"), &ctx)
            .await
            .unwrap();
        let nexus_claude::HookJSONOutput::Sync(sync) = out else {
            panic!("a sync answer")
        };
        assert_eq!(sync.reason.as_deref(), Some("skill context"));
        let t = timing_fields(clock.observe(&result("t1"), Utc::now()).unwrap());
        assert!(t["started_at"].is_f64(), "{t}");
    }

    #[tokio::test]
    async fn a_table_without_pre_tool_use_gets_the_clock_alone() {
        let clock = Arc::new(ToolClock::default());
        let mut table = super::super::agent_hooks::HookTable::new();
        clock_the_table(&mut table, Arc::clone(&clock));
        let ctx = nexus_claude::HookContext { signal: None };
        table["PreToolUse"][0].hooks[0]
            .execute(&pre_tool_use("Read"), Some("t2"), &ctx)
            .await
            .unwrap();
        let t = timing_fields(clock.observe(&result("t2"), Utc::now()).unwrap());
        assert!(t["started_at"].is_f64(), "{t}");
    }

    #[tokio::test]
    async fn the_legacy_tap_persists_and_relays_the_timing_of_a_result() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let sid = Uuid::new_v4();
        let (tx, mut rx) = broadcast::channel(16);
        spawn_legacy_tap(
            sid.to_string(),
            &tx,
            None,
            graph.clone(),
            Arc::new(AtomicI64::new(7)),
        );
        tx.send(tool_use("t1", "Read", json!({}))).unwrap();
        tx.send(result("t1")).unwrap();
        let relayed = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if let Ok(e @ ChatEvent::ToolTiming { .. }) = rx.recv().await {
                    return e;
                }
            }
        })
        .await
        .expect("a tool_timing on the channel");
        let t = timing_fields(relayed);
        assert_eq!(t["id"], "t1");
        let stored = graph.get_chat_events(sid, 0, 10).await.unwrap();
        assert_eq!(stored.len(), 1, "only the timing is the tap's to persist");
        assert_eq!(stored[0].event_type, "tool_timing");
        assert_eq!(stored[0].seq, 7);
    }

    #[test]
    fn the_hook_and_the_feeder_of_a_session_share_one_clock() {
        let sid = Uuid::new_v4().to_string();
        let first = ToolClock::for_session(&sid);
        assert!(Arc::ptr_eq(&first, &ToolClock::for_session(&sid)));
        let other = ToolClock::for_session(&Uuid::new_v4().to_string());
        assert!(!Arc::ptr_eq(&first, &other));
    }
}
